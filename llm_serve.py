"""Keep a model loaded on a Blythe GPU and answer prompts as Windmill adds them (a mailbox).

The llm_serve.sbatch job runs this in a run folder Windmill made:

    python llm_serve.py --models ~/narratrace/models <run folder>

input/request.json chooses the model and its settings, as for llm_batch.py, plus:
    "idle_minutes": 20,   # stop after this long with nothing to do (1 to 240)
    "poll_seconds": 5     # how often to look in inbox/ (1 to 60)
They are fixed for the whole job: a request can't change them.

Windmill adds work while the job runs by putting files in inbox/. Each file is written under
a hidden temporary name and then renamed, so the job never reads half a file:
    inbox/<request>.callback   optional, written first: an address to POST to once the
                               request is answered (https only)
    inbox/<request>.jsonl.gz   the prompts, one per line, in llm_batch.py's format:
                               {"id", "messages": [{"role", "content"}, ...], "labels", "meta"}
<request> is 1 to 64 letters, digits, '_' or '-'. Requests are answered one at a time,
oldest first. Anything else in inbox/ is left alone.

The answers go in outbox/:
    <request>.partial.jsonl.gz   answers so far, saved chunk by chunk
    <request>.jsonl.gz           every answer, in llm_batch.py's results format, once the
                                 request is finished
    <request>.json               the request's summary, written last: when it is there,
                                 the request is done
    <request>.error.json         instead of those, when the file isn't a valid request
Once a request is done (or refused), its files are removed from inbox/ and its callback is
called. Other files:
    status.json            what the job is doing now (rewritten when that changes, and at
                           least once a minute)
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

A request is data only: prompts and their labels. Nothing in it can choose the model, a file
or a command, and a file that isn't a valid request is refused with an error, not run.
"""

from __future__ import annotations

import argparse
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
    answer_chunk,
    append_results,
    engine_settings,
    gpu_details,
    load_model,
    log,
    read_done,
    read_prompts,
    rewrite_results,
    set_env,
)

SERVE_DEFAULTS = {"idle_minutes": 20, "poll_seconds": 5}
LIMITS = {"idle_minutes": (1, 240), "poll_seconds": (1, 60)}
REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PROMPTS_SUFFIX = ".jsonl.gz"
ROLES = {"system", "user", "assistant"}
MAX_LABELS, MAX_LABEL_CHARS = 100, 20
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


def check_prompts(prompts: list[dict]) -> None:
    """Every prompt must be plain chat messages with a short list of labels."""
    if not prompts:
        raise ValueError("the request has no prompts")
    for prompt in prompts:
        where = f"prompt {prompt.get('id')!r}"
        if not isinstance(prompt["id"], str) or not prompt["id"]:
            raise ValueError(f"{where}: id must be text")
        messages = prompt["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{where}: messages must be a list of chat messages")
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
        self.totals = {"requests": 0, "refused": 0, "prompts": 0, "prompt_tokens": 0}
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
            prompts = read_prompts(path)
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
        tokens = 0
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
            results = answer_chunk(
                self.llm, self.tokenizer, todo[start : start + size], self.request
            )
            append_results(partial, results)
            seconds += now() - t0
            tokens += sum(r["prompt_tokens"] or 0 for r in results)
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
            "n_answered": sum(1 for r in every if r["answer"] is not None),
            "n_no_answer": sum(1 for r in every if r["answer"] is None),
            "n_too_long": sum(1 for r in every if (r["error"] or "").startswith("prompt too long")),
            "prompts_sha256": prompts_sha256,
            "prompt_tokens_this_job": tokens,
            "seconds_generating": round(seconds, 1),
            "started": started,
            "finished": utc(),
        }
        write_json(self.outbox / f"{request_id}.json", body)
        self.totals["requests"] += 1
        self.totals["prompts"] += len(prompts)
        self.totals["prompt_tokens"] += tokens
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
    for name, (low, high) in LIMITS.items():
        value = request[name]
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"request.json {name} must be a whole number from {low} to {high}")
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
            "seconds_loading": round(seconds_loading, 1),
            "seconds_generating": round(server.seconds_generating, 1),
            "seconds_total": round(now() - started, 1),
            "left_in_inbox": waiting_requests(server.inbox),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
