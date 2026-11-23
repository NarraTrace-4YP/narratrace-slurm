"""Tests for llm_serve.py, the job that keeps a model loaded and answers the prompts Windmill
adds to its inbox while it runs.

As in test_llm_batch.py, a stand-in vLLM answers here (B for prompts mentioning "timescales",
else Z), and time is faked so the tests don't wait.
"""

import gzip
import http.server
import json
import os
import threading

import pytest
from test_llm_batch import LLM

import llm_batch
import llm_serve
from llm_serve import check_prompts, read_callback, waiting_requests


def make_run(tmp_path, **request):
    models = tmp_path / "models" / "Org" / "Model"
    models.mkdir(parents=True)
    (models / "config.json").write_text("{}")
    (models / "NARRATRACE_REVISION").write_text("abc123\n")
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    (run / "input" / "request.json").write_text(
        json.dumps({"model": "Org/Model", "chunk_size": 2, **request})
    )
    return tmp_path / "models", run


def add_request(run, request_id, contents, callback=None, labels=("A", "B", "Z")):
    """Adds a request the way Windmill does: callback first, then the prompts under a hidden
    name, renamed into place."""
    inbox = run / "inbox"
    inbox.mkdir(exist_ok=True)
    if callback:
        (inbox / f"{request_id}.callback").write_text(callback)
    temporary = inbox / f".{request_id}.jsonl.gz.tmp"
    with gzip.open(temporary, "wt") as f:
        for i, content in enumerate(contents):
            prompt = {
                "id": f"{request_id}-{i}",
                "messages": [{"role": "user", "content": content}],
                "labels": list(labels),
                "meta": {"n": i},
            }
            f.write(json.dumps(prompt) + "\n")
    temporary.replace(inbox / f"{request_id}.jsonl.gz")


def answers(run, request_id):
    return llm_batch.read_done(run / "outbox" / f"{request_id}.jsonl.gz")


def serve(models, run):
    return llm_serve.main(["--models", str(models), str(run)])


class Clock:
    """Fake time: each sleep moves the clock on, then runs the next thing a test planned
    (adding a request, asking it to stop...)."""

    def __init__(self, monkeypatch, *planned):
        self.t, self.planned = 1_000_000.0, list(planned)
        monkeypatch.setattr(llm_serve, "now", lambda: self.t)
        monkeypatch.setattr(llm_serve, "sleep", self.sleep)

    def sleep(self, seconds):
        self.t += seconds
        if self.planned:
            self.planned.pop(0)()


def test_answers_what_is_waiting_then_stops_when_asked(tmp_path, fake_vllm, monkeypatch):
    models, run = make_run(tmp_path)
    Clock(monkeypatch)
    add_request(run, "r1", ["about timescales", "about nothing", "timescales again"])
    (run / "stop").touch()
    assert serve(models, run) == 0

    got = answers(run, "r1")
    assert [r["id"] for r in got] == ["r1-0", "r1-1", "r1-2"]  # in the request's order
    assert [r["answer"] for r in got] == ["B", "Z", "B"] and got[0]["meta"] == {"n": 0}
    done = json.loads((run / "outbox" / "r1.json").read_text())
    assert done["status"] == "answered" and done["n_prompts"] == 3 and done["n_answered"] == 3
    assert len(done["prompts_sha256"]) == 64
    assert not list((run / "inbox").iterdir())  # taken out of the inbox
    assert not (run / "outbox" / "r1.partial.jsonl.gz").exists()
    summary = json.loads((run / "output" / "summary.json").read_text())
    assert summary["stopped_because"] == "asked to stop"
    assert summary["n_requests"] == 1 and summary["n_prompts"] == 3
    assert summary["model_revision"] == "abc123" and summary["left_in_inbox"] == []
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "stopped" and status["requests_done"] == 1
    record = (run / "output" / "requests.jsonl").read_text().splitlines()
    assert [json.loads(line)["request"] for line in record] == ["r1"]


def test_takes_requests_added_while_it_runs(tmp_path, fake_vllm, monkeypatch):
    models, run = make_run(tmp_path)
    Clock(
        monkeypatch,
        lambda: None,  # waits a while with nothing to do
        lambda: add_request(run, "first", ["about timescales"]),
        lambda: add_request(run, "second", ["about nothing", "about timescales"]),
        lambda: (run / "stop").touch(),
    )
    assert serve(models, run) == 0
    assert [r["answer"] for r in answers(run, "first")] == ["B"]
    assert [r["answer"] for r in answers(run, "second")] == ["Z", "B"]
    assert len(LLM.made) == 1  # the model was loaded once, for both requests
    record = (run / "output" / "requests.jsonl").read_text().splitlines()
    assert [json.loads(line)["request"] for line in record] == ["first", "second"]


def test_stops_by_itself_when_idle(tmp_path, fake_vllm, monkeypatch):
    models, run = make_run(tmp_path, idle_minutes=2, poll_seconds=30)
    clock = Clock(monkeypatch)
    assert serve(models, run) == 0
    summary = json.loads((run / "output" / "summary.json").read_text())
    assert summary["stopped_because"] == "idle for 2 min" and summary["n_requests"] == 0
    assert clock.t - 1_000_000.0 == 120  # four 30 s looks at the inbox
    assert json.loads((run / "status.json").read_text())["state"] == "stopped"


def test_time_limit_warning_then_a_requeue_carries_on(tmp_path, fake_vllm, monkeypatch):
    models, run = make_run(tmp_path)
    Clock(monkeypatch)
    add_request(run, "big", ["about timescales"] * 5)
    add_request(run, "later", ["about nothing"])
    real_answer_chunk = llm_serve.answer_chunk

    def warned_after_first_chunk(*args):
        llm_serve.stopping.set()  # as SIGUSR1 from SLURM would
        return real_answer_chunk(*args)

    monkeypatch.setattr(llm_serve, "answer_chunk", warned_after_first_chunk)
    assert serve(models, run) == 0
    summary = json.loads((run / "output" / "summary.json").read_text())
    assert summary["stopped_because"] == "time limit near"
    assert summary["left_in_inbox"] == ["big", "later"]
    partial = llm_batch.read_done(run / "outbox" / "big.partial.jsonl.gz")
    assert len(partial) == 2  # the chunk it was on was finished and saved

    # SLURM requeues the job in the same folder: it answers only what is left
    monkeypatch.setattr(llm_serve, "answer_chunk", real_answer_chunk)
    (run / "stop").touch()
    assert serve(models, run) == 0
    got = answers(run, "big")
    assert [r["id"] for r in got] == [f"big-{i}" for i in range(5)]
    assert [r["answer"] for r in answers(run, "later")] == ["Z"]
    done = json.loads((run / "outbox" / "big.json").read_text())
    assert done["prompt_tokens_this_job"] > 0 and done["n_prompts"] == 5


def test_a_bad_request_is_refused_and_the_rest_answered(tmp_path, fake_vllm, monkeypatch):
    models, run = make_run(tmp_path)
    Clock(monkeypatch)
    add_request(run, "bad", ["about timescales"], labels=[])
    add_request(run, "good", ["about timescales"])
    (run / "inbox" / "broken.jsonl.gz").write_bytes(b"not gzip at all")
    os.utime(run / "inbox" / "broken.jsonl.gz", ns=(1, 1))  # the oldest
    (run / "stop").touch()
    assert serve(models, run) == 0
    for request_id in ("bad", "broken"):
        error = json.loads((run / "outbox" / f"{request_id}.error.json").read_text())
        assert error["status"] == "refused" and error["error"]
        assert not (run / "outbox" / f"{request_id}.jsonl.gz").exists()
    assert [r["answer"] for r in answers(run, "good")] == ["B"]
    assert not list((run / "inbox").iterdir())
    summary = json.loads((run / "output" / "summary.json").read_text())
    assert summary["n_requests"] == 1 and summary["n_refused"] == 2


def test_only_request_files_are_taken(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("r1.jsonl.gz", "bad name.jsonl.gz", ".r2.jsonl.gz.tmp", "notes.txt", "r3.json"):
        (inbox / name).write_text("x")
    (inbox / "a.b.jsonl.gz").write_text("x")  # dots aren't allowed in a request id
    (inbox / "older.jsonl.gz").write_text("x")
    os.utime(inbox / "older.jsonl.gz", ns=(1, 1))
    assert waiting_requests(inbox) == ["older", "r1"]  # oldest first


def prompt(**changes):
    return {
        "id": "p",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "labels": ["A", "B"],
        **changes,
    }


def test_requests_are_data_only():
    check_prompts([prompt()])
    bad = [
        [],
        [prompt(id="")],
        [prompt(messages=[])],
        [prompt(messages=[{"role": "tool", "content": "x"}])],
        [prompt(messages=[{"role": "user", "content": "x", "name": "y"}])],
        [prompt(messages=[{"role": "user", "content": ["x"]}])],
        [prompt(labels=[])],
        [prompt(labels=["A", "A"])],
        [prompt(labels=["x" * 21])],
        [prompt(labels="AB")],
    ]
    for prompts in bad:
        with pytest.raises(ValueError):
            check_prompts(prompts)


def test_callback_addresses(tmp_path):
    path = tmp_path / "r.callback"
    assert read_callback(path) is None
    for url, ok in [
        ("https://windmill.rshields.xyz/api/w/dhr/jobs_u/resume/x", True),
        ("http://127.0.0.1:8000/x", True),
        ("http://windmill.rshields.xyz/x", False),
        ("file:///etc/passwd", False),
        ("https://a.example/x\nhttps://b.example/y", False),
    ]:
        path.write_text(url + "\n")
        assert (read_callback(path) == url) is ok, url


def test_callback_is_called_when_a_request_is_done(tmp_path, fake_vllm, monkeypatch):
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(201)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "1380001")
    try:
        models, run = make_run(tmp_path)
        Clock(monkeypatch)
        url = f"http://127.0.0.1:{httpd.server_port}/resume"
        add_request(run, "r1", ["about timescales"], callback=url)
        (run / "stop").touch()
        assert serve(models, run) == 0
    finally:
        httpd.shutdown()
    assert len(received) == 1
    assert received[0]["request"] == "r1" and received[0]["status"] == "answered"
    assert received[0]["slurm_job_id"] == "1380001"
    assert not (run / "inbox" / "r1.callback").exists()  # the address isn't left behind


@pytest.mark.parametrize(
    "setting",
    [
        {"idle_minutes": 0},
        {"idle_minutes": 241},
        {"poll_seconds": "5"},
        {"poll_seconds": True},
        {"engine": {"download_dir": "/tmp"}},
    ],
)
def test_settings_outside_their_limits_stop_the_job(tmp_path, fake_vllm, setting):
    models, run = make_run(tmp_path, **setting)
    assert serve(models, run) == 2
    assert not (run / "status.json").exists()  # stopped before loading anything


def test_a_missing_model_stops_the_job(tmp_path, fake_vllm):
    models, run = make_run(tmp_path, model="Org/Other")
    assert serve(models, run) == 2


def test_a_model_that_fails_to_load_says_so_in_the_status(tmp_path, fake_vllm, monkeypatch):
    class Broken:
        def __init__(self, **settings):
            raise ValueError("Free memory on device is less than desired GPU memory utilization")

    import sys

    monkeypatch.setattr(sys.modules["vllm"], "LLM", Broken)
    models, run = make_run(tmp_path)
    with pytest.raises(ValueError):
        serve(models, run)
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "failed" and "Free memory" in status["error"]
