"""Keep a model loaded on a Blythe GPU and answer prompts as Windmill adds them (a mailbox).

Fix (25 Nov 2026, problem review "llm_serve answers in rounds"):
    Problem: the job answered one request file at a time, in chunks, with the blocking
    llm.generate(). Each chunk waited for its slowest answer, and the platform's gateway
    sends many small files (one per batch of WeKnora calls), so the GPU only ever had a few
    prompts to work on: about 6 at once on Blythe, where vLLM's own server on a PC with the
    same model managed about 19.
    Why: llm.generate() takes a fixed list of prompts and returns when the last is done.
    Nothing new can join until then, and only one request file was read at a time.
    Solution: the job now drives vLLM's engine step by step (continuous batching, as vLLM's
    own server does). Every request file in inbox/ is taken as soon as it appears; its
    prompts are handed to the engine, a few from each request in turn, keeping about
    max_num_seqs of them running; each step writes one more token for every running prompt,
    and as soon as one finishes the next takes its place. A file's answers are written out
    when its last prompt finishes. The mailbox files, checks, limits and stop rules are the
    same; status.json now also says how many requests and prompts are in progress.

The llm_serve.sbatch job runs this in a run folder Windmill made:

    python llm_serve.py --models ~/narratrace/models <run folder>

input/request.json chooses the model and its settings, as for llm_batch.py, plus:
    "idle_minutes": 20,   # stop after this long with nothing to do (1 to 240)
    "poll_seconds": 1     # how often to look in inbox/ (0.2 to 60)
They are fixed for the whole job: a request can't change them.

Windmill adds work while the job runs by putting files in inbox/. Each file is written under
a hidden temporary name and then renamed, so the job never reads half a file:
    inbox/<request>.callback   optional, written first: an address to POST to once the
                               request is answered (https only)
    inbox/<request>.jsonl.gz   the prompts, one per line, each one of two kinds:
        a label prompt, as llm_batch.py's: the model answers with one of the labels
            {"id", "messages": [{"role", "content"}, ...], "labels": ["A", "B"], "meta"}
        a generation prompt: the model writes an answer
            {"id", "messages": [...], "generate": {...}, "tools": [...], "meta"}
        "generate" may set max_tokens (default 1024), temperature (0), top_p (1), stop (up
        to 8 strings), seed, json_schema (the answer must be JSON matching it) and thinking
        (false: Qwen's thinking block, kept apart as "reasoning"). "tools" (optional) lists
        functions in OpenAI's format; the chat template shows them to the model, and its
        calls come back in the answer's text (Qwen writes them as <tool_call> blocks).
        Generation prompts' messages may also have role "tool", tool_calls and
        tool_call_id, as in OpenAI's chat format.
<request> is 1 to 64 letters, digits, '_' or '-'. Every request waiting is worked on at
once (oldest first when there is a queue), sharing the GPU prompt by prompt. Anything else in
inbox/ is left alone.

The answers go in outbox/:
    <request>.partial.jsonl.gz   answers so far, saved every chunk_size answers (or 30 s)
    <request>.jsonl.gz           every answer, once the request is finished: for a label
                                 prompt llm_batch.py's results format; for a generation
                                 prompt {"id", "meta", "text", "reasoning", "finish_reason",
                                 "prompt_tokens", "completion_tokens", "error"}
    <request>.json               the request's summary, written last: when it is there,
                                 the request is done
    <request>.error.json         instead of those, when the file isn't a valid request
Once a request is done (or refused), its files are removed from inbox/ and its callback is
called. Other files:
    status.json            what the job is doing now (rewritten when that changes, and at
                           least once a minute)
    events.log             one line per finished or refused request, so whoever is waiting
                           can follow along without listing outbox/
    output/requests.jsonl  one line per finished request: the run's record
    output/summary.json    written at the end: versions, totals, and why it stopped

The job stops:
    - when a file named "stop" appears in the run folder, after answering everything
      already in inbox/;
    - when nothing has arrived for idle_minutes, so an idle job doesn't hold a shared GPU;
    - when SLURM warns that the time limit is near (SIGUSR1, see llm_serve.sbatch): it takes
      nothing new, gives the prompts already started up to 10 minutes to finish, saves
      their answers, and leaves the unfinished requests in inbox/.
If SLURM requeues the job in the same folder, it carries on: unfinished requests are still in
inbox/, and prompts already in a partial file aren't answered again.

A request is data only: chat messages and their settings, within fixed limits. Nothing in it
can choose the model, a file or a command, and a file that isn't a valid request is refused
with an error, not run. Tools are only descriptions for the model to read: nothing here runs
them; whoever sent the request decides what to do with the model's calls.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import signal
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import llm_batch
from llm_batch import (
    THINK_CLOSE,
    THINK_OPEN,
    append_results,
    close_thinking,
    engine_settings,
    gpu_details,
    load_model,
    log,
    pick_answer,
    read_done,
    rewrite_results,
    set_env,
)

SERVE_DEFAULTS = {"idle_minutes": 20, "poll_seconds": 1}
LIMITS = {"idle_minutes": (1, 240, int), "poll_seconds": (0.2, 60, float)}
REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PROMPTS_SUFFIX = ".jsonl.gz"
ROLES = {"system", "user", "assistant"}
GENERATE_ROLES = ROLES | {"tool"}
MAX_LABELS, MAX_LABEL_CHARS = 100, 20
# Generation settings a prompt may give, with their limits; nothing else is allowed.
DEFAULT_MAX_TOKENS, MAX_TOKENS = 1024, 32768
MAX_STOPS, MAX_STOP_CHARS = 8, 100
MAX_SCHEMA_BYTES, MAX_TOOLS, MAX_TOOLS_BYTES = 50_000, 64, 200_000
TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STATUS_EVERY = 60  # seconds: status.json is rewritten at least this often
STATUS_GAP = 2  # seconds: and at most this often while answering
SAVE_EVERY = 30  # seconds: answers are added to the partial files at least this often
STOP_GRACE = 600  # seconds the time limit warning leaves started prompts to finish
LOG_EVERY = 60  # seconds between progress lines in the log while answering
CALLBACK_TRIES, CALLBACK_GAP = 6, 20  # as the job files' own wake-up call

# Replaced in the tests, so they can move time on without waiting.
now = time.time
sleep = time.sleep

stopping = threading.Event()  # set by SIGUSR1: the time limit is near


def on_time_warning(signum, frame) -> None:
    stopping.set()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- files


def write_json(path: Path, data: dict) -> None:
    """Writes under a temporary name and renames, so a reader never sees half a file."""
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def waiting_requests(inbox: Path) -> list[str]:
    """The requests in inbox/, oldest first (by when the file was written, then by name).
    Hidden files (still being written) and names that aren't request ids are left alone."""
    found = []
    for path in inbox.glob("*" + PROMPTS_SUFFIX):
        request_id = path.name[: -len(PROMPTS_SUFFIX)]
        if REQUEST_ID.match(request_id) and path.is_file():
            try:
                found.append((path.stat().st_mtime_ns, request_id))
            except FileNotFoundError:  # removed while listing
                continue
    return [request_id for _, request_id in sorted(found)]


def read_request_prompts(path: Path) -> list[dict]:
    """A request file's prompts: one JSON object per line, each with an id (used once)."""
    prompts, seen = [], set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            prompt = json.loads(line)
            if not isinstance(prompt, dict) or "id" not in prompt or "messages" not in prompt:
                raise ValueError(f"line {n} isn't a prompt with an id and messages")
            if prompt["id"] in seen:
                raise ValueError(f"prompt id {prompt['id']!r} appears twice")
            seen.add(prompt["id"])
            prompts.append(prompt)
    return prompts


def is_generation(prompt: dict) -> bool:
    return "generate" in prompt


def _number(value, low, high, integer=False) -> bool:
    kinds = (int,) if integer else (int, float)
    return isinstance(value, kinds) and not isinstance(value, bool) and low <= value <= high


def check_generation(prompt: dict, where: str) -> None:
    """A generation prompt: OpenAI-style chat messages, settings within their limits, and
    optionally tools (descriptions only)."""
    unknown = sorted(set(prompt) - {"id", "messages", "generate", "tools", "meta"})
    if unknown:
        raise ValueError(f"{where}: unknown fields {unknown}")
    for message in prompt["messages"]:
        if not isinstance(message, dict) or message.get("role") not in GENERATE_ROLES:
            raise ValueError(f"{where}: each message needs a role, one of {GENERATE_ROLES}")
        extra = sorted(set(message) - {"role", "content", "tool_calls", "tool_call_id", "name"})
        if extra:
            raise ValueError(f"{where}: unknown message fields {extra}")
        content = message.get("content")
        calls = message.get("tool_calls")
        if calls is not None:
            if message["role"] != "assistant" or not isinstance(calls, list):
                raise ValueError(f"{where}: only assistant messages have a list of tool_calls")
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not TOOL_NAME.match(
                    str(function.get("name", ""))
                ):
                    raise ValueError(f"{where}: each tool call needs a function with a name")
                if not isinstance(function.get("arguments", ""), (str, dict)):
                    raise ValueError(f"{where}: tool call arguments must be text or an object")
        if not (isinstance(content, str) or (content is None and calls)):
            raise ValueError(f"{where}: message content must be text")
        for key in ("tool_call_id", "name"):
            if key in message and not isinstance(message[key], str):
                raise ValueError(f"{where}: {key} must be text")
    settings = prompt["generate"]
    if not isinstance(settings, dict):
        raise ValueError(f"{where}: generate must be an object of settings")
    allowed = {"max_tokens", "temperature", "top_p", "stop", "seed", "json_schema", "thinking"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown generate settings {unknown}")
    checks = {
        "max_tokens": lambda v: _number(v, 1, MAX_TOKENS, integer=True),
        "temperature": lambda v: _number(v, 0, 2),
        "top_p": lambda v: _number(v, 0.000001, 1),
        "seed": lambda v: v is None or _number(v, 0, 2**63 - 1, integer=True),
        "thinking": lambda v: isinstance(v, bool),
        "stop": lambda v: (
            isinstance(v, list)
            and len(v) <= MAX_STOPS
            and all(isinstance(s, str) and 0 < len(s) <= MAX_STOP_CHARS for s in v)
        ),
        "json_schema": lambda v: isinstance(v, dict) and len(json.dumps(v)) <= MAX_SCHEMA_BYTES,
    }
    for name, value in settings.items():
        if not checks[name](value):
            raise ValueError(f"{where}: generate setting {name} is out of range: {value!r}")
    if settings.get("thinking") and "json_schema" in settings:
        raise ValueError(f"{where}: json_schema can't be used with thinking")
    tools = prompt.get("tools")
    if tools is not None:
        if (
            not isinstance(tools, list)
            or len(tools) > MAX_TOOLS
            or len(json.dumps(tools)) > MAX_TOOLS_BYTES
        ):
            raise ValueError(f"{where}: tools must be a list of at most {MAX_TOOLS} functions")
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if (
                tool.get("type") != "function"
                or not isinstance(function, dict)
                or not TOOL_NAME.match(str(function.get("name", "")))
                or not isinstance(function.get("description", ""), str)
                or not isinstance(function.get("parameters", {}), dict)
            ):
                raise ValueError(
                    f"{where}: each tool is {{type: function, function: {{name, description,"
                    " parameters}}}}"
                )


def check_prompts(prompts: list[dict]) -> None:
    """Every prompt must be plain chat messages, with either a short list of labels or
    generation settings within their limits."""
    if not prompts:
        raise ValueError("the request has no prompts")
    for prompt in prompts:
        where = f"prompt {prompt.get('id')!r}"
        if not isinstance(prompt["id"], str) or not prompt["id"]:
            raise ValueError(f"{where}: id must be text")
        messages = prompt["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{where}: messages must be a list of chat messages")
        if ("labels" in prompt) == is_generation(prompt):
            raise ValueError(f"{where}: give either labels or generate, not both or neither")
        if is_generation(prompt):
            check_generation(prompt, where)
            continue
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in ROLES
                or not isinstance(message["content"], str)
            ):
                raise ValueError(f"{where}: each message is {{role, content}}, role one of {ROLES}")
        labels = prompt["labels"]
        if (
            not isinstance(labels, list)
            or not 1 <= len(labels) <= MAX_LABELS
            or len(set(labels)) != len(labels)
            or not all(isinstance(x, str) and 0 < len(x) <= MAX_LABEL_CHARS for x in labels)
        ):
            raise ValueError(
                f"{where}: labels must be 1 to {MAX_LABELS} different short strings"
                f" (up to {MAX_LABEL_CHARS} characters)"
            )


def read_callback(path: Path) -> str | None:
    """The request's callback address, if it left one and it is https (http only to this
    machine, for the tests)."""
    if not path.exists():
        return None
    url = path.read_text().strip()
    parsed = urlparse(url)
    local = parsed.hostname in ("127.0.0.1", "localhost")
    if "\n" in url or len(url) > 2000 or not parsed.hostname:
        return None
    if parsed.scheme == "https" or (parsed.scheme == "http" and local):
        return url
    return None


def call_back(url: str, body: dict) -> None:
    """POSTs body to url, trying up to CALLBACK_TRIES times, CALLBACK_GAP s apart."""
    data = json.dumps(body).encode()
    for attempt in range(1, CALLBACK_TRIES + 1):
        try:
            call = urllib.request.Request(
                url, data=data, method="POST", headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(call, timeout=20):
                pass
            log(f"Called back for request {body.get('request')} (attempt {attempt})")
            return
        except Exception as error:  # noqa: BLE001 - Windmill checks later anyway
            if attempt == CALLBACK_TRIES:
                log(f"Couldn't call back for request {body.get('request')}: {error}")
                return
            sleep(CALLBACK_GAP)


# ---------------------------------------------------------------- building and reading prompts


def structured_output(schema: dict) -> dict:
    """SamplingParams' setting that makes the answer JSON matching schema (vLLM renamed it)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}


def final_only(params):
    """Asks vLLM to hand back an answer only once it is finished, not after every token."""
    try:
        from vllm.sampling_params import RequestOutputKind

        params.output_kind = RequestOutputKind.FINAL_ONLY
    except (ImportError, AttributeError):  # an older vLLM, or the tests' stand-in
        pass
    return params


def split_thinking(text: str) -> tuple[str | None, str]:
    """(reasoning, answer) from a reply that may start with a thinking block."""
    if THINK_CLOSE not in text:
        return None, text
    reasoning, _, answer = text.partition(THINK_CLOSE)
    return reasoning.replace(THINK_OPEN, "").strip(), answer.lstrip()


def empty_result(prompt: dict) -> dict:
    """A prompt's result before the model has answered, in the format of its kind."""
    if is_generation(prompt):
        return {
            "id": prompt["id"],
            "meta": prompt.get("meta"),
            "text": None,
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": None,
        }
    return {
        "id": prompt["id"],
        "meta": prompt.get("meta"),
        "answer": None,
        "probabilities": {},
        "label_mass": 0.0,
        "prompt_tokens": None,
        "error": None,
    }


def build(tokenizer, prompt: dict, request: dict, has_think: bool):
    """(result, token ids, SamplingParams) for one prompt, as llm_batch.answer_chunk builds a
    label prompt (one token, with its likeliest alternatives) and generate settings a
    generation prompt. Token ids are None when the prompt can't be run: the result then
    says why."""
    from vllm import SamplingParams

    result = empty_result(prompt)
    try:
        if is_generation(prompt):
            settings = prompt["generate"]
            thinking = bool(settings.get("thinking"))
            options = {"tools": prompt["tools"]} if prompt.get("tools") else {}
            text = tokenizer.apply_chat_template(
                prompt["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
                **options,
            )
            if not thinking:
                text = close_thinking(text, has_think)
        else:
            text = tokenizer.apply_chat_template(
                prompt["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            text = close_thinking(text, has_think)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception as error:  # noqa: BLE001 - one bad prompt mustn't stop the others
        result["error"] = f"couldn't build the prompt: {error}"
        return result, None, None
    result["prompt_tokens"] = len(token_ids)
    room = request["max_model_len"] - len(token_ids)
    if room < 1:
        result["error"] = f"prompt too long ({len(token_ids)} tokens)"
        return result, None, None
    if not is_generation(prompt):
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=request["top_logprobs"])
        return result, token_ids, final_only(params)
    try:
        settings = prompt["generate"]
        params = SamplingParams(
            max_tokens=min(settings.get("max_tokens", DEFAULT_MAX_TOKENS), room),
            temperature=settings.get("temperature", 0.0),
            top_p=settings.get("top_p", 1.0),
            stop=settings.get("stop") or None,
            seed=settings.get("seed"),
            **(structured_output(settings["json_schema"]) if "json_schema" in settings else {}),
        )
    except Exception as error:  # noqa: BLE001
        result["error"] = f"couldn't build the prompt: {error}"
        return result, None, None
    return result, token_ids, final_only(params)


def read_output(tokenizer, prompt: dict, result: dict, output) -> None:
    """Fills result in from vLLM's finished output for prompt."""
    out = output.outputs[0]
    if is_generation(prompt):
        thinking = bool(prompt["generate"].get("thinking"))
        reasoning, text = split_thinking(out.text) if thinking else (None, out.text)
        result.update(
            text=text,
            reasoning=reasoning,
            finish_reason=out.finish_reason,
            completion_tokens=len(out.token_ids),
        )
        return
    first = out.logprobs[0] if out.logprobs else {}
    top = {}
    for token_id, logprob in first.items():
        token = logprob.decoded_token
        if token is None:
            token = tokenizer.decode([token_id])
        top[token] = max(top.get(token, -math.inf), logprob.logprob)
    answer, probabilities, mass = pick_answer(top, prompt["labels"])
    result.update(answer=answer, probabilities=probabilities, label_mass=mass)
    if answer is None:
        result["error"] = "no label among the model's likeliest first tokens"


def answered(result: dict) -> bool:
    return result.get("answer") is not None or result.get("text") is not None


# ---------------------------------------------------------------- the server


class Job:
    """One request file being answered: its prompts, the answers so far, and the prompts
    not yet given to the model."""

    def __init__(self, request_id: str, prompts: list[dict], sha256: str, partial: Path):
        self.request_id, self.prompts, self.sha256, self.partial = (
            request_id,
            prompts,
            sha256,
            partial,
        )
        self.results: dict[str, dict] = {}  # prompt id -> result, in any order
        self.queue: list[dict] = []  # prompts not yet given to the model
        self.in_flight = 0  # prompts the model is working on
        self.unsaved: list[dict] = []  # results not yet in the partial file
        self.saved_at = now()
        self.started, self.first_submit = utc(), None
        self.prompt_tokens = self.completion_tokens = 0

    def done(self) -> bool:
        return not self.queue and not self.in_flight

    def add(self, result: dict) -> None:
        self.results[result["id"]] = result
        self.unsaved.append(result)
        self.prompt_tokens += result["prompt_tokens"] or 0
        self.completion_tokens += result.get("completion_tokens") or 0

    def save(self) -> None:
        """Adds the results not yet saved to the partial file, so a job that is stopped
        doesn't answer them again."""
        if self.unsaved:
            append_results(self.partial, self.unsaved)
            self.unsaved = []
        self.saved_at = now()


class Server:
    """Keeps the model busy: every request in inbox/ is taken as soon as it arrives, and its
    prompts are handed to vLLM's engine a few at a time, so the engine always has about
    max_num_seqs prompts to work on together. Each step of the engine writes one more token
    of every prompt it is working on; a prompt that finishes makes room for the next at once
    (continuous batching), so short and long answers, and prompts from different requests,
    share the GPU instead of waiting for each other."""

    def __init__(self, folder: Path, request: dict, llm, tokenizer, record: dict):
        self.folder, self.request, self.llm, self.tokenizer = folder, request, llm, tokenizer
        self.inbox, self.outbox = folder / "inbox", folder / "outbox"
        self.output = folder / "output"
        for path in (self.inbox, self.outbox, self.output):
            path.mkdir(parents=True, exist_ok=True)
        self.record = record  # goes into output/summary.json
        self.callbacks: list[threading.Thread] = []
        self.status: dict = {}
        self.status_written = 0.0
        self.totals = {
            "requests": 0,
            "refused": 0,
            "prompts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self.seconds_generating = 0.0
        self.jobs: dict[str, Job] = {}  # requests being answered, in the order they were taken
        self.engine_ids: dict[str, tuple[Job, dict, dict]] = {}  # vLLM request id -> its prompt
        self.counter = 0
        self.has_think = False
        self.most_in_flight = 0
        self.logged_at = now()

    @property
    def engine(self):
        return self.llm.llm_engine

    @property
    def in_flight(self) -> int:
        return len(self.engine_ids)

    def room(self) -> int:
        """How many prompts the engine is given at once: max_num_seqs, plus a few waiting so
        a finished prompt's place is filled straight away."""
        most = int(self.request["max_num_seqs"])
        return most + max(1, most // 4)

    # --- status.json

    def set_status(self, state: str, force: bool = False, **details) -> None:
        """Rewrites status.json when the state changes, when its details change (at most every
        STATUS_GAP seconds) and at least every STATUS_EVERY seconds."""
        status = {"state": state, **details}
        if not force and state == self.status.get("state"):
            gap = STATUS_EVERY if status == self.status else STATUS_GAP
            if now() - self.status_written < gap:
                return
        self.status, self.status_written = status, now()
        write_json(
            self.folder / "status.json",
            {
                **status,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "model": self.request["model"],
                "requests_done": self.totals["requests"],
                "updated": utc(),
            },
        )

    def answering_status(self) -> None:
        jobs = list(self.jobs.values())
        self.set_status(
            "answering",
            requests=[job.request_id for job in jobs][:20],
            n_requests=len(jobs),
            prompts_in_flight=self.in_flight,
            answered=sum(len(job.results) for job in jobs),
            of=sum(len(job.prompts) for job in jobs),
        )

    # --- finishing or refusing a request

    def finish(self, request_id: str, body: dict) -> None:
        """Removes the request from inbox/ and calls its callback (in the background)."""
        prompts = self.inbox / f"{request_id}{PROMPTS_SUFFIX}"
        callback = self.inbox / f"{request_id}.callback"
        url = read_callback(callback)
        prompts.unlink(missing_ok=True)
        callback.unlink(missing_ok=True)
        with open(self.output / "requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
        event = {"request": request_id, "status": body["status"], "at": utc()}
        with open(self.folder / "events.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        if url:
            call = threading.Thread(
                target=call_back,
                args=(url, {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), **body}),
                daemon=True,
            )
            call.start()
            self.callbacks.append(call)

    def refuse(self, request_id: str, error: Exception) -> None:
        body = {"request": request_id, "status": "refused", "error": str(error), "at": utc()}
        write_json(self.outbox / f"{request_id}.error.json", body)
        self.totals["refused"] += 1
        log(f"Refused request {request_id}: {error}")
        self.finish(request_id, body)

    def take(self, request_id: str) -> None:
        """Starts answering a request in inbox/: checks it, and queues the prompts that the
        partial file (from before a requeue) doesn't already answer."""
        path = self.inbox / f"{request_id}{PROMPTS_SUFFIX}"
        try:
            raw = path.read_bytes()
            prompts = read_request_prompts(path)
            check_prompts(prompts)
        except FileNotFoundError:
            return  # removed before it could be read
        except Exception as error:  # noqa: BLE001 - a bad file is refused, not fatal
            self.refuse(request_id, error)
            return
        partial = self.outbox / f"{request_id}.partial{PROMPTS_SUFFIX}"
        job = Job(request_id, prompts, hashlib.sha256(raw).hexdigest(), partial)
        ids = {p["id"] for p in prompts}
        done = [r for r in read_done(partial) if r["id"] in ids]
        if done:
            rewrite_results(partial, done)
        job.results = {r["id"]: r for r in done}
        job.queue = [p for p in prompts if p["id"] not in job.results]
        self.jobs[request_id] = job
        log(f"Request {request_id}: {len(prompts)} prompt(s), {len(job.queue)} to answer")

    def complete(self, job: Job) -> None:
        """Writes a finished request's answers and summary, and lets whoever sent it know."""
        job.save()
        del self.jobs[job.request_id]
        by_id = {r["id"]: r for r in read_done(job.partial)}
        missing = [p["id"] for p in job.prompts if p["id"] not in by_id]
        if missing:
            raise RuntimeError(f"request {job.request_id}: {len(missing)} answers weren't saved")
        final = self.outbox / f"{job.request_id}{PROMPTS_SUFFIX}"
        rewrite_results(final, [by_id[p["id"]] for p in job.prompts])
        job.partial.unlink(missing_ok=True)
        every = list(by_id.values())
        seconds = now() - job.first_submit if job.first_submit is not None else 0.0
        body = {
            "request": job.request_id,
            "status": "answered",
            "n_prompts": len(job.prompts),
            "n_answered": sum(1 for r in every if answered(r)),
            "n_no_answer": sum(1 for r in every if not answered(r)),
            "n_too_long": sum(1 for r in every if (r["error"] or "").startswith("prompt too long")),
            "prompts_sha256": job.sha256,
            "prompt_tokens_this_job": job.prompt_tokens,
            "completion_tokens_this_job": job.completion_tokens,
            "seconds_generating": round(seconds, 1),  # from its first prompt to its last
            "started": job.started,
            "finished": utc(),
        }
        write_json(self.outbox / f"{job.request_id}.json", body)
        self.totals["requests"] += 1
        self.totals["prompts"] += len(job.prompts)
        self.totals["prompt_tokens"] += job.prompt_tokens
        self.totals["completion_tokens"] += job.completion_tokens
        log(f"Request {job.request_id}: done, {body['n_answered']} of {len(job.prompts)} answered")
        self.finish(job.request_id, body)

    # --- feeding the engine

    def top_up(self) -> None:
        """Gives the engine prompts until it has room() of them, taking one from each request
        in turn, so a short request isn't stuck behind a long one."""
        while self.in_flight < self.room():
            jobs = [job for job in self.jobs.values() if job.queue]
            if not jobs:
                return
            for job in jobs:
                if self.in_flight >= self.room():
                    return
                self.submit(job, job.queue.pop(0))

    def submit(self, job: Job, prompt: dict) -> None:
        result, token_ids, params = build(self.tokenizer, prompt, self.request, self.has_think)
        if token_ids is None:
            job.add(result)  # too long, or couldn't be built: answered with its error
            return
        self.counter += 1
        engine_id = f"{job.request_id}/{self.counter}"
        try:
            self.engine.add_request(engine_id, {"prompt_token_ids": token_ids}, params)
        except Exception as error:  # noqa: BLE001 - vLLM refused this prompt (a schema it
            # can't follow, say): it gets an error, and the others are still answered
            result["error"] = f"the model couldn't answer: {error}"
            job.add(result)
            return
        if job.first_submit is None:
            job.first_submit = now()
        job.in_flight += 1
        self.engine_ids[engine_id] = (job, prompt, result)
        self.most_in_flight = max(self.most_in_flight, self.in_flight)

    def step(self) -> None:
        """Runs one step of the engine and records the prompts it finished."""
        t0 = now()
        outputs = self.engine.step()
        self.seconds_generating += now() - t0
        for output in outputs:
            if not getattr(output, "finished", True) or output.request_id not in self.engine_ids:
                continue
            job, prompt, result = self.engine_ids.pop(output.request_id)
            job.in_flight -= 1
            try:
                read_output(self.tokenizer, prompt, result, output)
            except Exception as error:  # noqa: BLE001
                result["error"] = f"couldn't read the model's answer: {error}"
            job.add(result)

    def save_and_complete(self, final: bool = False) -> None:
        """Saves answers to the partial files every chunk_size answers (or SAVE_EVERY
        seconds), and writes out each request that is finished."""
        size = max(1, int(self.request["chunk_size"]))
        for job in list(self.jobs.values()):
            if job.done() and not final:
                self.complete(job)
            elif job.unsaved and (
                final or len(job.unsaved) >= size or now() - job.saved_at >= SAVE_EVERY
            ):
                job.save()

    def stop_engine(self) -> None:
        """Called when the time limit is near: lets the prompts in the engine finish (for up
        to STOP_GRACE seconds), saves their answers, and drops the rest. The unfinished
        requests stay in inbox/ for a requeued job."""
        if self.in_flight:
            log(f"Time limit near: finishing the {self.in_flight} prompt(s) already started")
        deadline = now() + STOP_GRACE
        while self.in_flight and now() < deadline:
            self.step()
            self.save_and_complete()
        if self.in_flight:
            log(f"Dropping {self.in_flight} unfinished prompt(s); a requeued job redoes them")
            self.engine.abort_request(list(self.engine_ids))
            self.engine_ids.clear()
        for job in list(self.jobs.values()):
            if job.done():
                self.complete(job)
        self.save_and_complete(final=True)

    def log_progress(self) -> None:
        if now() - self.logged_at >= LOG_EVERY:
            self.logged_at = now()
            log(
                f"{len(self.jobs)} request(s) open, {self.in_flight} prompt(s) in the engine,"
                f" {self.totals['requests']} request(s) done"
            )

    # --- the loop

    def run(self) -> str:
        """Answers requests until it is told to stop or has been idle too long; returns why
        it stopped."""
        self.has_think = THINK_OPEN in self.tokenizer.get_vocab()
        idle_seconds = self.request["idle_minutes"] * 60
        last_work = now()
        looked = None  # when inbox/ was last listed
        while True:
            if stopping.is_set():
                self.stop_engine()
                return "time limit near"
            busy = bool(self.jobs)
            # Look for new requests every poll_seconds (each step only takes a moment).
            if looked is None or not busy or now() - looked >= self.request["poll_seconds"]:
                looked = now()
                for request_id in waiting_requests(self.inbox):
                    if request_id not in self.jobs:
                        self.take(request_id)
                self.save_and_complete()  # a request whose prompts were all done already
            if self.jobs:
                self.top_up()
                if self.in_flight:
                    self.step()
                self.save_and_complete()
                self.answering_status()
                self.log_progress()
                last_work = now()
                continue
            if (self.folder / "stop").exists():
                return "asked to stop"
            idle = now() - last_work
            if idle >= idle_seconds:
                return f"idle for {self.request['idle_minutes']} min"
            self.set_status("waiting", idle_minutes_left=round((idle_seconds - idle) / 60))
            sleep(self.request["poll_seconds"])

    def wait_for_callbacks(self) -> None:
        deadline = now() + CALLBACK_TRIES * (CALLBACK_GAP + 20)
        for call in self.callbacks:
            call.join(timeout=max(0.0, deadline - now()))


# ---------------------------------------------------------------- starting


def read_request(folder: Path) -> dict:
    """input/request.json with every default filled in; refuses settings outside their limits."""
    request = {
        **llm_batch.DEFAULTS,
        **SERVE_DEFAULTS,
        **json.loads((folder / "input" / "request.json").read_text()),
    }
    for name, (low, high, kind) in LIMITS.items():
        if not _number(request[name], low, high, integer=kind is int):
            raise ValueError(f"request.json {name} must be a number from {low} to {high}")
    engine_settings(request)
    return request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True, help="folder of downloaded models")
    parser.add_argument("folder", type=Path, help="the run folder")
    args = parser.parse_args(argv)

    started = now()
    stopping.clear()
    signal.signal(signal.SIGUSR1, on_time_warning)
    folder = args.folder
    try:
        request = read_request(folder)
        set_env(request)
    except (ValueError, OSError) as error:
        log(f"ERROR {error}")
        return 2
    model_dir = args.models / request["model"]
    if not (model_dir / "config.json").exists():
        log(f"ERROR {request['model']} isn't downloaded to {model_dir}: run setup_env.sbatch")
        return 2

    import torch
    import vllm

    revision = model_dir / "NARRATRACE_REVISION"
    record = {
        "model": request["model"],
        "model_revision": revision.read_text().strip() if revision.exists() else None,
        "request": request,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "node": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "code_commit": os.environ.get("NARRATRACE_CODE_COMMIT"),  # this repository's
        **gpu_details(),
        "started": utc(),
    }
    server = Server(folder, request, None, None, record)
    server.set_status("loading", force=True)
    try:
        t0 = now()
        server.llm = load_model(model_dir, request)
        server.tokenizer = server.llm.get_tokenizer()
        seconds_loading = now() - t0
        log(f"Model loaded in {seconds_loading:.0f} s; waiting for requests in {server.inbox}")
        reason = server.run()
    except BaseException as error:
        # Windmill reads status.json, so say what went wrong there too; the log has the rest.
        server.set_status("failed", force=True, error=f"{type(error).__name__}: {error}"[:2000])
        raise
    log(f"Stopping: {reason}")
    server.set_status("stopped", force=True, reason=reason)
    server.wait_for_callbacks()
    write_json(
        server.output / "summary.json",
        {
            **record,
            "finished": utc(),
            "stopped_because": reason,
            "n_requests": server.totals["requests"],
            "n_refused": server.totals["refused"],
            "n_prompts": server.totals["prompts"],
            "prompt_tokens": server.totals["prompt_tokens"],
            "completion_tokens": server.totals["completion_tokens"],
            "seconds_loading": round(seconds_loading, 1),
            "seconds_generating": round(server.seconds_generating, 1),
            "seconds_total": round(now() - started, 1),
            "left_in_inbox": waiting_requests(server.inbox),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
