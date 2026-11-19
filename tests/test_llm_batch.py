"""Tests for llm_batch.py, the job that answers prompts with vLLM on Blythe.

vLLM and a GPU are only on Blythe, so a stand-in vLLM gives made-up first-token
probabilities here; the real model is checked by running the pilot on Blythe.
"""

import gzip
import json
import math
import sys
import types

import pytest

import llm_batch
from llm_batch import close_thinking, pick_answer


def test_pick_answer_takes_the_likeliest_label():
    top = {"B": math.log(0.6), " A": math.log(0.3), "The": math.log(0.05)}
    answer, probabilities, mass = pick_answer(top, ["A", "B", "Z"])
    assert answer == "B" and probabilities == {"A": 0.3, "B": 0.6} and mass == 0.9


def test_pick_answer_adds_up_spellings_and_breaks_ties_by_order():
    top = {"A": math.log(0.2), " A": math.log(0.25), "B": math.log(0.4)}
    assert pick_answer(top, ["A", "B"])[0] == "A"  # 0.45 for A, both spellings together
    top = {"A": math.log(0.2), " A": math.log(0.2), "B": math.log(0.4)}
    assert pick_answer(top, ["A", "B"])[0] == "A"  # a tie: the earlier label
    assert pick_answer(top, ["B", "A"])[0] == "B"


def test_pick_answer_without_any_label():
    assert pick_answer({"The": -0.1, "<think>": -2.0}, ["A", "Z"]) == (None, {}, 0.0)


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("...assistant\n<think>\n\n</think>\n\n", "...assistant\n<think>\n\n</think>\n\n"),
        ("...assistant\n<think>\n\n</think>", "...assistant\n<think>\n\n</think>\n\n"),
        ("...assistant\n<think>\n", "...assistant\n<think>\n\n</think>\n\n"),
        ("...assistant\n", "...assistant\n<think>\n\n</think>\n\n"),
    ],
)
def test_close_thinking(prompt, expected):
    assert close_thinking(prompt, True) == expected


def test_close_thinking_leaves_models_without_it_alone():
    assert close_thinking("...assistant\n", False) == "...assistant\n"


# ---------------------------------------------------------------- the whole job, with a stand-in vLLM


class Logprob:
    def __init__(self, probability, token):
        self.logprob, self.decoded_token = math.log(probability), token


class Tokenizer:
    def get_vocab(self):
        return {"<think>": 0, "</think>": 1, "A": 2, "B": 3, "Z": 4}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "<user>" + messages[-1]["content"] + "</user><assistant>\n"

    def __call__(self, text, add_special_tokens):
        assert text.endswith("</think>\n\n")
        return {"input_ids": text.split()}

    def decode(self, ids):
        return "?"


class LLM:
    made = []

    def __init__(self, **settings):
        if (settings.get("limit_mm_per_prompt") or {}).get("video") == 0:
            raise ValueError("this model has no video modality")
        LLM.made.append(settings)

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, prompts, params, use_tqdm):
        assert params.max_tokens == 1 and params.temperature == 0
        outputs = []
        for prompt in prompts:
            # the stand-in answers B for prompts mentioning "timescales", else "none of these"
            words = " ".join(prompt["prompt_token_ids"])
            if "timescales" in words:
                first = {3: Logprob(0.8, "B"), 2: Logprob(0.15, " A"), 9: Logprob(0.05, "The")}
            elif "gibberish" in words:
                first = {9: Logprob(0.9, "The")}
            else:
                first = {4: Logprob(0.7, "Z"), 2: Logprob(0.2, "A")}
            outputs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[first])]))
        return outputs


class SamplingParams:
    def __init__(self, max_tokens, temperature, logprobs):
        self.max_tokens, self.temperature, self.logprobs = max_tokens, temperature, logprobs


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = LLM, SamplingParams, "0.34.0"
    torch = types.ModuleType("torch")
    torch.__version__, torch.version = "2.14.0+cu130", types.SimpleNamespace(cuda="13.0")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    LLM.made = []


def write_run(tmp_path, prompts, **request):
    models = tmp_path / "models" / "Org" / "Model"
    models.mkdir(parents=True)
    (models / "config.json").write_text("{}")
    (models / "NARRATRACE_REVISION").write_text("abc123\n")
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    (run / "input" / "request.json").write_text(
        json.dumps({"model": "Org/Model", "chunk_size": 2, **request})
    )
    with gzip.open(run / "input" / "prompts.jsonl.gz", "wt") as f:
        for i, content in enumerate(prompts):
            f.write(
                json.dumps(
                    {
                        "id": f"p{i}",
                        "messages": [{"role": "user", "content": content}],
                        "labels": ["A", "B", "Z"],
                        "meta": {"n": i},
                    }
                )
                + "\n"
            )
    return tmp_path / "models", run


def results(run):
    return llm_batch.read_done(run / "output" / "results.jsonl.gz")


def test_job_answers_saves_and_summarises(tmp_path, fake_vllm):
    models, run = write_run(
        tmp_path,
        ["about timescales", "about nothing", "gibberish here", "word " * 50],
        max_model_len=40,
    )
    code = llm_batch.main(["--models", str(models), str(run / "input"), str(run / "output")])
    assert code == 0
    by_id = {r["id"]: r for r in results(run)}
    assert by_id["p0"]["answer"] == "B" and by_id["p0"]["probabilities"] == {"A": 0.15, "B": 0.8}
    assert by_id["p0"]["meta"] == {"n": 0}
    assert by_id["p1"]["answer"] == "Z"
    assert by_id["p2"]["answer"] is None and "no label" in by_id["p2"]["error"]
    assert by_id["p3"]["answer"] is None and by_id["p3"]["error"].startswith("prompt too long")
    summary = json.loads((run / "output" / "summary.json").read_text())
    assert summary["n_prompts"] == summary["n_results"] == 4
    assert summary["n_answered"] == 2 and summary["n_too_long"] == 1
    assert summary["model_revision"] == "abc123" and summary["vllm"] == "0.34.0"
    assert summary["cuda"] == "13.0" and len(summary["prompts_sha256"]) == 64
    # it fell back from the video limit the model refused to the image limit
    assert LLM.made[-1]["limit_mm_per_prompt"] == {"image": 0}
    assert LLM.made[-1]["max_num_seqs"] == 128  # vLLM's 256 is too many for Qwen3.8 on an L40


def test_job_carries_on_where_it_stopped(tmp_path, fake_vllm):
    models, run = write_run(tmp_path, ["about timescales"] * 5)
    llm_batch.main(["--models", str(models), str(run / "input"), str(run / "output")])
    first = results(run)
    # a job killed while writing leaves a cut-off end: the complete lines are kept
    path = run / "output" / "results.jsonl.gz"
    whole = path.read_bytes()
    cut = next(
        n for n in range(10, len(whole), 10) if len(llm_batch.read_done_bytes(whole[:-n])) < 5
    )
    path.write_bytes(whole[:-cut])
    kept = len(results(run))
    assert 0 < kept < 5
    llm_batch.main(["--models", str(models), str(run / "input"), str(run / "output")])
    again = results(run)
    assert sorted(r["id"] for r in again) == sorted(r["id"] for r in first)


def test_job_stops_if_the_model_is_missing(tmp_path, fake_vllm):
    models, run = write_run(tmp_path, ["x"], model="Org/Other")
    assert llm_batch.main(["--models", str(models), str(run / "input"), str(run / "output")]) == 2


def test_prompts_need_unique_ids(tmp_path):
    path = tmp_path / "p.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for _ in range(2):
            f.write(json.dumps({"id": "same", "messages": [], "labels": ["A"]}) + "\n")
    with pytest.raises(ValueError, match="twice"):
        llm_batch.read_prompts(path)
