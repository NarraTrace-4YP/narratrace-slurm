"""Answer a batch of multiple-choice prompts with an LLM, offline with vLLM, on one GPU.

The llm_batch.sbatch job runs this on a Blythe GPU node, inside the run folder Windmill made:

    python llm_batch.py --models ~/narratrace/models input output

input/request.json says how to run the model:
    {"model": "Qwen/Qwen3.8-27B-FP8",  # a folder under --models, made by setup_env.sbatch
     "max_model_len": 8192,            # longer prompts are reported, not answered
     "top_logprobs": 20,               # how many of the model's likeliest first tokens to read
     "chunk_size": 512,                # prompts per vLLM call; results are saved after each
     "gpu_memory_utilization": 0.9,
     "max_num_seqs": 128,              # prompts worked on at once (see DEFAULTS)
     "seed": 0}
input/prompts.jsonl.gz holds one prompt per line:
    {"id": "...", "messages": [{"role": "system" | "user", "content": "..."}, ...],
     "labels": ["A", "B", ..., "Z"], "meta": {...}}

Each prompt asks for one label (a letter). The model is not left to write freely: it is
asked for a single token, and its answer is the label it gives the most probability among
its likeliest first tokens. The probability of every label it lists is kept, so a doubtful
answer can be told from a sure one. Thinking (Qwen's <think> block) is switched off, and an
empty thinking block is written into the prompt if the chat template doesn't do it, so the
first token is the answer.

output/results.jsonl.gz gets one line per prompt, added chunk by chunk, so a job that runs
out of time still leaves every finished chunk behind:
    {"id", "meta",            # copied from the prompt
     "answer": "B" | null,    # null: no label among the likeliest tokens, or prompt too long
     "probabilities": {"A": 0.01, "B": 0.97},  # labels among the likeliest tokens
     "label_mass": 0.98,      # how much of the probability went to labels at all
     "prompt_tokens": 1234, "error": null | "..."}
output/summary.json: counts, timings, and the versions that produced the answers (model
revision, this repository's commit, vLLM, torch, CUDA, GPU and driver).

Running again in the same folder (for example after SLURM requeued the job) answers only the
prompts not yet in results.jsonl.gz.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS = {
    "max_model_len": 8192,
    "top_logprobs": 20,
    "chunk_size": 512,
    "gpu_memory_utilization": 0.9,
    # How many prompts vLLM works on at once. Qwen3.8 is a hybrid (Mamba) model: each prompt
    # in flight needs a Mamba cache block, and an L40 has room for about 208 (Blythe,
    # 19 Nov 2026), so vLLM's default of 256 fails at start-up.
    "max_num_seqs": 128,
    "seed": 0,
}
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {message}", flush=True)


# ---------------------------------------------------------------- reading answers


def pick_answer(top_tokens: dict[str, float], labels: list[str]):
    """Turns the model's likeliest first tokens into an answer.

    top_tokens maps each decoded token to its log-probability. Tokens are compared without
    surrounding spaces, so " B" and "B" both count for label B. Returns (answer,
    probabilities, label_mass): the label with the most probability (the earlier label on a
    tie), each label's probability, and their total. No label among the tokens: (None, {}, 0).
    """
    probabilities: dict[str, float] = {}
    for token, logprob in top_tokens.items():
        label = token.strip()
        if label in labels:
            probabilities[label] = probabilities.get(label, 0.0) + math.exp(logprob)
    if not probabilities:
        return None, {}, 0.0
    order = {label: i for i, label in enumerate(labels)}
    answer = max(probabilities, key=lambda label: (probabilities[label], -order[label]))
    rounded = {
        label: round(probabilities[label], 4) for label in sorted(probabilities, key=order.get)
    }
    return answer, rounded, round(sum(probabilities.values()), 4)


def close_thinking(prompt: str, has_think_tokens: bool) -> str:
    """Makes sure a rendered chat prompt ends with an empty, closed thinking block.

    With thinking switched off, Qwen's chat template already ends the prompt with
    "<think>\\n\\n</think>\\n\\n". If a template ignores the switch (or only opens the block),
    the block is completed here, so the model's first token is its answer. Models without
    thinking tokens are left alone."""
    if not has_think_tokens:
        return prompt
    stripped = prompt.rstrip()
    if stripped.endswith(THINK_CLOSE):
        return stripped + "\n\n"
    if stripped.endswith(THINK_OPEN):
        return stripped + "\n\n" + THINK_CLOSE + "\n\n"
    return prompt + THINK_OPEN + "\n\n" + THINK_CLOSE + "\n\n"


# ---------------------------------------------------------------- files


def read_prompts(path: Path) -> list[dict]:
    prompts, seen = [], set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            prompt = json.loads(line)
            for key in ("id", "messages", "labels"):
                if key not in prompt:
                    raise ValueError(f"prompt on line {n} has no {key!r}")
            if prompt["id"] in seen:
                raise ValueError(f"prompt id {prompt['id']!r} appears twice")
            seen.add(prompt["id"])
            prompts.append(prompt)
    return prompts


def gunzip_complete(raw: bytes) -> bytes:
    """Unzips a file of gzip members, keeping whatever a cut-off last member still holds."""
    out, data = bytearray(), raw
    while data:
        member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out += member.decompress(data)
        except zlib.error:
            break
        if not member.eof:
            break
        data = member.unused_data
    return bytes(out)


def read_done_bytes(raw: bytes) -> list[dict]:
    done = []
    for line in gunzip_complete(raw).decode("utf-8", errors="replace").splitlines():
        try:
            done.append(json.loads(line))
        except ValueError:
            break
    return done


def read_done(path: Path) -> list[dict]:
    """The results already written, keeping only complete lines (a job killed while writing
    can leave a cut-off line or gzip member at the end)."""
    return read_done_bytes(path.read_bytes()) if path.exists() else []


def rewrite_results(path: Path, results: list[dict]) -> None:
    """Writes results afresh (dropping a cut-off end), so more can be added after them."""
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_results(path: Path, results: list[dict]) -> None:
    """Adds results as a new gzip member and closes the file, so every chunk is saved."""
    with gzip.open(path, "at", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_details() -> dict:
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        name, driver, memory = [part.strip() for part in out[0].split(",")]
        return {"gpu": name, "driver": driver, "gpu_memory": memory}
    except Exception as error:  # noqa: BLE001 - only for the record
        return {"gpu": f"unknown ({error})"}


# ---------------------------------------------------------------- running the model


def load_model(model_dir: Path, request: dict):
    from vllm import LLM

    settings = dict(
        model=str(model_dir),
        max_model_len=request["max_model_len"],
        gpu_memory_utilization=request["gpu_memory_utilization"],
        max_num_seqs=request["max_num_seqs"],
        seed=request["seed"],
        max_logprobs=max(20, request["top_logprobs"]),
    )
    # The prompts are text only: tell a vision-language model to reserve no memory for
    # images or video. Older vLLM versions or text-only models may reject a modality.
    for limits in ({"image": 0, "video": 0}, {"image": 0}, None):
        try:
            if limits is None:
                return LLM(**settings)
            return LLM(**settings, limit_mm_per_prompt=limits)
        except (TypeError, ValueError) as error:
            if limits is None:
                raise
            log(f"vLLM didn't take limit_mm_per_prompt={limits} ({error}); trying without")


def answer_chunk(llm, tokenizer, prompts: list[dict], request: dict) -> list[dict]:
    from vllm import SamplingParams

    has_think = THINK_OPEN in tokenizer.get_vocab()
    results, to_run = [], []
    for prompt in prompts:
        result = {
            "id": prompt["id"],
            "meta": prompt.get("meta"),
            "answer": None,
            "probabilities": {},
            "label_mass": 0.0,
            "prompt_tokens": None,
            "error": None,
        }
        try:
            text = tokenizer.apply_chat_template(
                prompt["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_ids = tokenizer(close_thinking(text, has_think), add_special_tokens=False)
            token_ids = token_ids["input_ids"]
            result["prompt_tokens"] = len(token_ids)
            if len(token_ids) >= request["max_model_len"]:
                result["error"] = f"prompt too long ({len(token_ids)} tokens)"
            else:
                to_run.append((result, token_ids, prompt["labels"]))
        except Exception as error:  # noqa: BLE001 - one bad prompt mustn't stop the batch
            result["error"] = f"couldn't build the prompt: {error}"
        results.append(result)

    if to_run:
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=request["top_logprobs"])
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for _, ids, _ in to_run], params, use_tqdm=False
        )
        for (result, _, labels), output in zip(to_run, outputs, strict=True):
            first = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
            top = {}
            for token_id, logprob in first.items():
                token = logprob.decoded_token
                if token is None:
                    token = tokenizer.decode([token_id])
                top[token] = max(top.get(token, -math.inf), logprob.logprob)
            answer, probabilities, mass = pick_answer(top, labels)
            result.update(answer=answer, probabilities=probabilities, label_mass=mass)
            if answer is None:
                result["error"] = "no label among the model's likeliest first tokens"
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True, help="folder of downloaded models")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    started = time.time()
    request = {**DEFAULTS, **json.loads((args.input / "request.json").read_text())}
    model_dir = args.models / request["model"]
    if not (model_dir / "config.json").exists():
        log(f"ERROR {request['model']} isn't downloaded to {model_dir}: run setup_env.sbatch")
        return 2
    prompts_path = args.input / "prompts.jsonl.gz"
    prompts = read_prompts(prompts_path)
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.jsonl.gz"
    done = read_done(results_path)
    if done:
        rewrite_results(results_path, done)
    done_ids = {r["id"] for r in done}
    todo = [p for p in prompts if p["id"] not in done_ids]
    log(f"{len(prompts)} prompts, {len(done_ids)} already answered, {len(todo)} to go")

    import torch
    import vllm

    summary = {
        "model": request["model"],
        "model_revision": (model_dir / "NARRATRACE_REVISION").read_text().strip()
        if (model_dir / "NARRATRACE_REVISION").exists()
        else None,
        "request": request,
        "prompts_sha256": sha256_of(prompts_path),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "node": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "code_commit": os.environ.get("NARRATRACE_CODE_COMMIT"),  # this repository's
        **gpu_details(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    seconds_loading = seconds_generating = 0.0
    prompt_tokens = 0
    if todo:
        t0 = time.time()
        llm = load_model(model_dir, request)
        tokenizer = llm.get_tokenizer()
        seconds_loading = time.time() - t0
        log(f"Model loaded in {seconds_loading:.0f} s")
        size = max(1, int(request["chunk_size"]))
        for start in range(0, len(todo), size):
            t0 = time.time()
            results = answer_chunk(llm, tokenizer, todo[start : start + size], request)
            append_results(results_path, results)
            elapsed = time.time() - t0
            seconds_generating += elapsed
            tokens = sum(r["prompt_tokens"] or 0 for r in results)
            prompt_tokens += tokens
            log(
                f"{min(start + size, len(todo))}/{len(todo)} answered"
                f" ({tokens / max(elapsed, 1e-9):.0f} prompt tokens/s)"
            )

    every = read_done(results_path)
    summary.update(
        finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_prompts=len(prompts),
        n_results=len(every),
        n_answered=sum(1 for r in every if r["answer"] is not None),
        n_no_answer=sum(1 for r in every if r["answer"] is None),
        n_too_long=sum(1 for r in every if (r["error"] or "").startswith("prompt too long")),
        prompt_tokens_this_run=prompt_tokens,
        seconds_loading=round(seconds_loading, 1),
        seconds_generating=round(seconds_generating, 1),
        prompt_tokens_per_second=round(prompt_tokens / seconds_generating, 1)
        if seconds_generating
        else None,
        seconds_total=round(time.time() - started, 1),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"Done: {summary['n_answered']} answered, {summary['n_no_answer']} without an answer")
    return 0 if summary["n_results"] == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
