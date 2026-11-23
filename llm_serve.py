"""Keep a model loaded on a Blythe GPU and answer prompts as Windmill adds them (a mailbox).

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
<request> is 1 to 64 letters, digits, '_' or '-'. Requests are answered one at a time,
oldest first. Anything else in inbox/ is left alone.

The answers go in outbox/:
    <request>.partial.jsonl.gz   answers so far, saved chunk by chunk
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
    - when SLURM warns that the time limit is near (SIGUSR1, see llm_serve.sbatch): it
      finishes the chunk it is on and leaves the rest of its requests in inbox/.
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
    answer_chunk,
    append_results,
    close_thinking,
    engine_settings,
    gpu_details,
    load_model,
    log,
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
STATUS_EVERY = 60  # seconds: status.json is rewritten at least this often while waiting
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


# ---------------------------------------------------------------- writing answers


def structured_output(schema: dict) -> dict:
    """SamplingParams' setting that makes the answer JSON matching schema (vLLM renamed it)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}


def split_thinking(text: str) -> tuple[str | None, str]:
    """(reasoning, answer) from a reply that may start with a thinking block."""
    if THINK_CLOSE not in text:
        return None, text
    reasoning, _, answer = text.partition(THINK_CLOSE)
    return reasoning.replace(THINK_OPEN, "").strip(), answer.lstrip()


def generate_chunk(llm, tokenizer, prompts: list[dict], request: dict) -> list[dict]:
    """Writes an answer to each generation prompt. One prompt that vLLM refuses (a schema it
    can't follow, say) gets an error; the others are still answered."""
    from vllm import SamplingParams

    has_think = THINK_OPEN in tokenizer.get_vocab()
    results, to_run = [], []
    for prompt in prompts:
        settings = prompt["generate"]
        thinking = bool(settings.get("thinking"))
        result = {
            "id": prompt["id"],
            "meta": prompt.get("meta"),
            "text": None,
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": None,
        }
        try:
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
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            result["prompt_tokens"] = len(token_ids)
            room = request["max_model_len"] - len(token_ids)
            if room < 1:
                result["error"] = f"prompt too long ({len(token_ids)} tokens)"
            else:
                params = SamplingParams(
                    max_tokens=min(settings.get("max_tokens", DEFAULT_MAX_TOKENS), room),
                    temperature=settings.get("temperature", 0.0),
                    top_p=settings.get("top_p", 1.0),
                    stop=settings.get("stop") or None,
                    seed=settings.get("seed"),
                    **(
                        structured_output(settings["json_schema"])
                        if "json_schema" in settings
                        else {}
                    ),
                )
                to_run.append((result, token_ids, params, thinking))
        except Exception as error:  # noqa: BLE001 - one bad prompt mustn't stop the batch
            result["error"] = f"couldn't build the prompt: {error}"
        results.append(result)

    def run(batch):
        return llm.generate(
            [{"prompt_token_ids": ids} for _, ids, _, _ in batch],
            [params for _, _, params, _ in batch],
            use_tqdm=False,
        )

    def record(item, output) -> None:
        result, _, _, thinking = item
        out = output.outputs[0]
        reasoning, text = split_thinking(out.text) if thinking else (None, out.text)
        result.update(
            text=text,
            reasoning=reasoning,
            finish_reason=out.finish_reason,
            completion_tokens=len(out.token_ids),
        )

    if to_run:
        try:
            for item, output in zip(to_run, run(to_run), strict=True):
                record(item, output)
        except Exception:  # noqa: BLE001 - find the prompt vLLM refused, answer the rest
            for item in to_run:
                try:
                    record(item, run([item])[0])
                except Exception as error:  # noqa: BLE001
                    item[0]["error"] = f"the model couldn't answer: {error}"
    return results


def answered(result: dict) -> bool:
    return result.get("answer") is not None or result.get("text") is not None


# ---------------------------------------------------------------- the server


class Server:
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

    # --- status.json

    def set_status(self, state: str, force: bool = False, **details) -> None:
        status = {"state": state, **details}
        if not force and status == self.status and now() - self.status_written < STATUS_EVERY:
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

    # --- one request

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

    def answer(self, request_id: str) -> bool:
        """Answers one request. False if the time limit warning stopped it part way (the
        request stays in inbox/, its answers so far in the partial file)."""
        path = self.inbox / f"{request_id}{PROMPTS_SUFFIX}"
        try:
            prompts_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            prompts = read_request_prompts(path)
            check_prompts(prompts)
        except FileNotFoundError:
            return True  # removed before it could be read
        except Exception as error:  # noqa: BLE001 - a bad file is refused, not fatal
            self.refuse(request_id, error)
            return True

        started = utc()
        partial = self.outbox / f"{request_id}.partial{PROMPTS_SUFFIX}"
        ids = {p["id"] for p in prompts}
        done = [r for r in read_done(partial) if r["id"] in ids]
        if done:
            rewrite_results(partial, done)
        done_ids = {r["id"] for r in done}
        todo = [p for p in prompts if p["id"] not in done_ids]
        log(f"Request {request_id}: {len(prompts)} prompt(s), {len(todo)} to answer")
        size = max(1, int(self.request["chunk_size"]))
        tokens = completion_tokens = 0
        seconds = 0.0
        for start in range(0, len(todo), size):
            self.set_status(
                "answering",
                force=True,
                request=request_id,
                answered=len(done_ids) + start,
                of=len(prompts),
            )
            t0 = now()
            chunk = todo[start : start + size]
            labelled = [p for p in chunk if not is_generation(p)]
            generated = [p for p in chunk if is_generation(p)]
            results = []
            if labelled:
                results += answer_chunk(self.llm, self.tokenizer, labelled, self.request)
            if generated:
                results += generate_chunk(self.llm, self.tokenizer, generated, self.request)
            append_results(partial, results)
            seconds += now() - t0
            tokens += sum(r["prompt_tokens"] or 0 for r in results)
            completion_tokens += sum(r.get("completion_tokens") or 0 for r in results)
            if stopping.is_set() and start + size < len(todo):
                self.seconds_generating += seconds
                log(f"Request {request_id}: stopped part way for the time limit")
                return False

        by_id = {r["id"]: r for r in read_done(partial)}
        missing = [p["id"] for p in prompts if p["id"] not in by_id]
        if missing:
            raise RuntimeError(f"request {request_id}: {len(missing)} answers weren't saved")
        final = self.outbox / f"{request_id}{PROMPTS_SUFFIX}"
        rewrite_results(final, [by_id[p["id"]] for p in prompts])
        partial.unlink(missing_ok=True)
        every = list(by_id.values())
        body = {
            "request": request_id,
            "status": "answered",
            "n_prompts": len(prompts),
            "n_answered": sum(1 for r in every if answered(r)),
            "n_no_answer": sum(1 for r in every if not answered(r)),
            "n_too_long": sum(1 for r in every if (r["error"] or "").startswith("prompt too long")),
            "prompts_sha256": prompts_sha256,
            "prompt_tokens_this_job": tokens,
            "completion_tokens_this_job": completion_tokens,
            "seconds_generating": round(seconds, 1),
            "started": started,
            "finished": utc(),
        }
        write_json(self.outbox / f"{request_id}.json", body)
        self.totals["requests"] += 1
        self.totals["prompts"] += len(prompts)
        self.totals["prompt_tokens"] += tokens
        self.totals["completion_tokens"] += completion_tokens
        self.seconds_generating += seconds
        log(f"Request {request_id}: done, {body['n_answered']} of {len(prompts)} answered")
        self.finish(request_id, body)
        return True

    # --- the loop

    def run(self) -> str:
        """Answers requests until it is told to stop or has been idle too long; returns why
        it stopped."""
        idle_seconds = self.request["idle_minutes"] * 60
        last_work = now()
        while True:
            if stopping.is_set():
                return "time limit near"
            waiting = waiting_requests(self.inbox)
            if waiting:
                if not self.answer(waiting[0]):
                    return "time limit near"
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
