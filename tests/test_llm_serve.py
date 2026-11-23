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
from types import SimpleNamespace

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


# ---------------------------------------------------------------- generation prompts


class Params:
    """Stands in for vllm.SamplingParams: keeps whatever it is given."""

    def __init__(self, **settings):
        self.__dict__.update(settings)


class Structured:
    def __init__(self, json):
        self.json = json


class TextTokenizer:
    def get_vocab(self):
        return {"<think>": 0, "</think>": 1, "A": 2, "B": 3, "Z": 4}

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking, tools=None
    ):
        text = " ".join(m.get("content") or "" for m in messages)
        if tools:
            text += " tools: " + " ".join(t["function"]["name"] for t in tools)
        return text + (" <think>\n" if enable_thinking else "\n")

    def __call__(self, text, add_special_tokens):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return "?"


class TextLLM:
    """A stand-in model that writes predictable answers: JSON under a schema, a tool call when
    it was shown tools, a thinking block when thinking is on, and an echo otherwise. A prompt
    mentioning "refuse" makes the whole batch fail, as vLLM does with a schema it can't use."""

    made = []

    def __init__(self, **settings):
        TextLLM.made.append(settings)

    def get_tokenizer(self):
        return TextTokenizer()

    def generate(self, prompts, params, use_tqdm):
        if not isinstance(params, list):
            params = [params] * len(prompts)
        outputs = []
        for prompt, p in zip(prompts, params, strict=True):
            words = prompt["prompt_token_ids"]
            if p.max_tokens == 1:  # a label prompt, as llm_batch asks
                first = {3: Logprob(0.9, "B")} if "timescales" in words else {4: Logprob(0.8, "Z")}
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[first])]))
                continue
            if "refuse" in words:
                raise ValueError("this schema can't be followed")
            if getattr(p, "structured_outputs", None):
                text = json.dumps({"ok": True, "schema_keys": sorted(p.structured_outputs.json)})
            elif "tools:" in words:
                text = '<tool_call>\n{"name": "lookup", "arguments": {"q": "x"}}\n</tool_call>'
            elif words[-1] == "<think>":
                text = "<think>\nweighing it up\n</think>\n\nThe answer."
            else:
                said = [w for w in words if w not in ("<think>", "</think>")]
                text = "echo " + " ".join(said[:3])
            text = " ".join(text.split(" ")[: p.max_tokens]) if p.max_tokens < 5 else text
            outputs.append(
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=text,
                            finish_reason="length" if p.max_tokens < 5 else "stop",
                            token_ids=list(range(len(text.split()))),
                        )
                    ]
                )
            )
        return outputs


class Logprob:
    def __init__(self, probability, token):
        import math

        self.logprob, self.decoded_token = math.log(probability), token


@pytest.fixture
def text_vllm(monkeypatch):
    import sys
    import types

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = TextLLM, Params, "0.34.0"
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.StructuredOutputsParams = Structured
    torch = types.ModuleType("torch")
    torch.__version__, torch.version = "2.14.0+cu130", types.SimpleNamespace(cuda="13.0")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    monkeypatch.setitem(sys.modules, "torch", torch)
    TextLLM.made = []


def add_raw_request(run, request_id, prompts):
    inbox = run / "inbox"
    inbox.mkdir(exist_ok=True)
    temporary = inbox / f".{request_id}.jsonl.gz.tmp"
    with gzip.open(temporary, "wt") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
    temporary.replace(inbox / f"{request_id}.jsonl.gz")


def user(text):
    return [{"role": "user", "content": text}]


def test_generation_prompts_get_written_answers(tmp_path, text_vllm, monkeypatch):
    models, run = make_run(tmp_path)
    Clock(monkeypatch)
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    history = [
        {"role": "user", "content": "find x"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "nothing found"},
    ]
    add_raw_request(
        run,
        "mixed",
        [
            {"id": "label", "messages": user("about timescales"), "labels": ["A", "B"]},
            {"id": "plain", "messages": user("hello there model"), "generate": {}},
            {
                "id": "json",
                "messages": user("give json"),
                "generate": {"json_schema": {"type": "object", "properties": {}}},
            },
            {"id": "think", "messages": user("hard one"), "generate": {"thinking": True}},
            {"id": "tool", "messages": history, "generate": {"max_tokens": 200}, "tools": [tool]},
            {"id": "short", "messages": user("say a lot"), "generate": {"max_tokens": 2}},
        ],
    )
    (run / "stop").touch()
    assert serve(models, run) == 0
    got = {r["id"]: r for r in answers(run, "mixed")}
    assert [r["id"] for r in answers(run, "mixed")] == [
        "label",
        "plain",
        "json",
        "think",
        "tool",
        "short",
    ]
    assert got["label"]["answer"] == "B"
    assert (
        got["plain"]["text"] == "echo hello there model" and got["plain"]["finish_reason"] == "stop"
    )
    assert got["plain"]["completion_tokens"] == 4 and got["plain"]["prompt_tokens"] > 0
    assert json.loads(got["json"]["text"]) == {"ok": True, "schema_keys": ["properties", "type"]}
    assert got["think"]["reasoning"] == "weighing it up" and got["think"]["text"] == "The answer."
    assert got["tool"]["text"].startswith("<tool_call>") and got["tool"]["error"] is None
    assert got["short"]["finish_reason"] == "length"
    done = json.loads((run / "outbox" / "mixed.json").read_text())
    assert done["n_answered"] == 6 and done["completion_tokens_this_job"] > 0
    events = [json.loads(line) for line in (run / "events.log").read_text().splitlines()]
    assert [(e["request"], e["status"]) for e in events] == [("mixed", "answered")]
    summary = json.loads((run / "output" / "summary.json").read_text())
    assert summary["completion_tokens"] == done["completion_tokens_this_job"]


def test_a_prompt_the_model_refuses_gets_an_error_and_the_rest_are_answered(
    tmp_path, text_vllm, monkeypatch
):
    models, run = make_run(tmp_path, chunk_size=10)
    Clock(monkeypatch)
    add_raw_request(
        run,
        "r",
        [
            {"id": "a", "messages": user("first"), "generate": {}},
            {"id": "b", "messages": user("please refuse"), "generate": {}},
            {"id": "c", "messages": user("third"), "generate": {}},
        ],
    )
    (run / "stop").touch()
    assert serve(models, run) == 0
    got = {r["id"]: r for r in answers(run, "r")}
    assert got["a"]["text"] == "echo first" and got["c"]["text"] == "echo third"
    assert got["b"]["text"] is None and "can't be followed" in got["b"]["error"]


def test_a_prompt_longer_than_the_model_allows_is_reported(tmp_path, text_vllm, monkeypatch):
    models, run = make_run(tmp_path, max_model_len=10)
    Clock(monkeypatch)
    add_raw_request(run, "r", [{"id": "long", "messages": user("word " * 20), "generate": {}}])
    (run / "stop").touch()
    assert serve(models, run) == 0
    (got,) = answers(run, "r")
    assert got["text"] is None and got["error"].startswith("prompt too long")


def generation(**changes):
    prompt = {"id": "g", "messages": user("hi"), "generate": {}}
    prompt.update(changes)
    return prompt


@pytest.mark.parametrize(
    "prompt",
    [
        generation(generate={"model": "other"}),
        generation(generate={"max_tokens": 0}),
        generation(generate={"max_tokens": 40000}),
        generation(generate={"temperature": 3}),
        generation(generate={"top_p": 0}),
        generation(generate={"stop": ["x"] * 9}),
        generation(generate={"seed": -1}),
        generation(generate={"json_schema": "not an object"}),
        generation(generate={"json_schema": {"type": "object"}, "thinking": True}),
        generation(generate="max_tokens=5"),
        generation(tools=[{"type": "function", "function": {"name": "rm -rf ~"}}]),
        generation(tools=[{"type": "shell", "command": "ls"}]),
        generation(tools="lookup"),
        generation(model="other"),
        generation(messages=[{"role": "system", "content": "x", "command": "ls"}]),
        generation(messages=[{"role": "root", "content": "x"}]),
        generation(messages=[{"role": "user", "content": None}]),
        generation(messages=[{"role": "user", "content": "x", "tool_calls": []}]),
        {"id": "g", "messages": user("hi"), "generate": {}, "labels": ["A"]},
        {"id": "g", "messages": user("hi")},
        {"id": "g", "messages": [{"role": "tool", "content": "x"}], "labels": ["A"]},
    ],
)
def test_generation_requests_are_data_only(prompt):
    with pytest.raises(ValueError):
        check_prompts([prompt])


def test_generation_settings_within_limits_pass():
    check_prompts(
        [
            generation(
                generate={
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["\n\n"],
                    "seed": 7,
                    "json_schema": {"type": "object"},
                },
                tools=[{"type": "function", "function": {"name": "lookup", "description": "d"}}],
                meta={"page": "Police"},
            )
        ]
    )


def test_polling_can_be_quicker_than_a_second(tmp_path, text_vllm, monkeypatch):
    models, run = make_run(tmp_path, poll_seconds=0.5, idle_minutes=1)
    clock = Clock(monkeypatch)
    assert serve(models, run) == 0
    assert clock.t - 1_000_000.0 == 60
    models, run = make_run(tmp_path / "again", poll_seconds=0.1)
    assert serve(models, run) == 2
