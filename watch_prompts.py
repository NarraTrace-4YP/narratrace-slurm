"""Watch the prompts going in and the answers coming out of a running llm_serve job, live.

Run it on Blythe, in a second terminal, while the job runs (it only reads files):

    ~/narratrace/env/bin/python ~/narratrace/slurm/watch_prompts.py                 # your newest gateway job
    ~/narratrace/env/bin/python ~/narratrace/slurm/watch_prompts.py <run folder>    # a particular one
    ~/narratrace/env/bin/python ~/narratrace/slurm/watch_prompts.py --full          # whole prompts, not just the end

Each request file Windmill or the gateway drops in inbox/ is copied as soon as it appears
(the job deletes it once answered) and its prompts are printed; each answer file that
appears in outbox/ is printed too. Ctrl-C to stop. Plain Python, nothing to install.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

RUNS = Path.home() / "narratrace" / "runs"
GREY, BLUE, GREEN, RED, OFF = "\033[90m", "\033[94m", "\033[92m", "\033[91m", "\033[0m"


def newest_run() -> Path:
    runs = sorted(RUNS.glob("gw-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        sys.exit(f"No gateway run folders in {RUNS}; give the run folder instead.")
    return runs[0]


def lines(raw: bytes) -> list[dict]:
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, EOFError):
        return []  # still being written
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else "…" + text[-n:]


def show_prompts(name: str, prompts: list[dict], full: bool) -> None:
    print(f"\n{BLUE}▶ {time.strftime('%H:%M:%S')} {name}: {len(prompts)} prompt(s) in{OFF}")
    for p in prompts:
        kind = "labels " + "/".join(p["labels"]) if "labels" in p else "write"
        print(f"{GREY}  [{p['id']}] {kind}{OFF}")
        for m in p.get("messages", []):
            content = m.get("content") or ""
            if m.get("tool_calls"):
                content += " " + json.dumps(m["tool_calls"])
            body = content if full else short(content, 400)
            print(f"    {m.get('role', '?')}: {body}")


def show_answers(name: str, answers: list[dict], full: bool) -> None:
    print(f"\n{GREEN}◀ {time.strftime('%H:%M:%S')} {name}: {len(answers)} answer(s) out{OFF}")
    for a in answers:
        if a.get("error"):
            print(f"  [{a['id']}] {RED}error: {a['error']}{OFF}")
        elif "answer" in a:
            print(f"  [{a['id']}] {a['answer']}  {GREY}{a.get('probabilities')}{OFF}")
        else:
            text = a.get("text") or ""
            body = text if full else short(text, 600)
            tokens = a.get("completion_tokens")
            print(f"  [{a['id']}] {GREY}({tokens} tokens, {a.get('finish_reason')}){OFF} {body}")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv
    run = Path(args[0]).expanduser() if args else newest_run()
    inbox, outbox = run / "inbox", run / "outbox"
    print(f"Watching {run} (Ctrl-C to stop)")
    seen_in: set[str] = set()
    seen_out = {p.name for p in outbox.glob("*.jsonl.gz")}  # only what arrives from now on
    while True:
        for path in sorted(inbox.glob("*.jsonl.gz")):
            if path.name in seen_in or path.name.startswith("."):
                continue
            try:
                prompts = lines(path.read_bytes())
            except FileNotFoundError:
                continue  # answered and removed before it could be read
            if prompts:
                seen_in.add(path.name)
                show_prompts(path.name[: -len(".jsonl.gz")], prompts, full)
        for path in sorted(outbox.glob("*.jsonl.gz")):
            if path.name in seen_out or ".partial." in path.name:
                continue
            answers = lines(path.read_bytes())
            if answers:
                seen_out.add(path.name)
                show_answers(path.name[: -len(".jsonl.gz")], answers, full)
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
