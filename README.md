# NarraTrace SLURM jobs

The fixed jobs NarraTrace runs on Blythe, the University of Warwick's GPU cluster (SCRTP).
The platform (`NarraTrace-4YP/4yp_platform`) submits them from Windmill. The first user is
O1 stage 5, `f/dhr/O1/choose_template_section`, which runs an LLM to map each review onto
its statutory template.

This repository is kept apart from the platform because it is what runs on Blythe, under
each team member's own university account. Each person has their own clone on Blythe, which
changes only when they pull it there. The platform's Windmill code changes separately.

## How Windmill uses it

Each Windmill run logs in over SSH as the person who started it, with their own key and a 2FA
code made from their own 2FA secret, which they keep as their own Windmill variables (the
platform's `f/dhr/README.md` lists them). SCRTP allowed this for Rowan's account on 19 Nov
2026, until they offer a web interface; anyone else needs the same agreement for their own
account first. Windmill can only:

- upload a run's input files into a new run folder,
- submit one of the job files in `jobs/` by name, with a time limit,
- ask SLURM for the job's state, cancel the job, and download its output.

It never sends commands of its own. The job names Windmill may submit are listed in
`f/dhr/blythe.py` (`JOBS`), so adding a job here means adding its name there too.

| Job file | What it does |
| --- | --- |
| `jobs/setup_env.sbatch` | Installs uv, Python 3.12 and vLLM 0.34.0 into `~/narratrace/env`, and downloads the model at a fixed commit. Needs no GPU. Run it by hand once, and again after changing a version. |
| `jobs/test_gpu.sbatch` | Ten minutes on one L40. It reports the GPU and its driver, then calls Windmill back. |
| `jobs/llm_batch.sbatch` | Answers a batch of one-letter prompts with vLLM on one L40 (`llm_batch.py`; its docstring describes the input and output files). |
| `jobs/llm_serve.sbatch` | Keeps a model loaded on one L40 (or split over more when Windmill asks for them: the gateway runs full-precision Qwen3.8-27B on two, and the job first checks the GPUs can talk to each other with `check_gpus.py`) and answers prompts that Windmill adds while it runs, with a label or with written text, JSON or tool calls (`llm_serve.py`; see "Adding prompts to a running job" below). The platform's LLM gateway uses it. |

Each job's last step POSTs to the address in its run folder's `callback_url` (up to six
tries, 20 s apart). This wakes the waiting Windmill run. Windmill then asks SLURM whether
the job has really ended, so a lost or early call does no harm.

`llm_batch` records this repository's commit in `output/summary.json` (`code_commit`,
ending in `-dirty` if the clone has uncommitted changes). The platform stores that file with
the answers, so every result can be traced to the code that produced it.

## Adding prompts to a running job

`llm_batch` is given all its prompts at the start. `llm_serve` loads the model once and then
answers prompts as Windmill adds them, so more work doesn't mean waiting in the SLURM queue
or loading the model again. Its run folder works as a mailbox:

```text
runs/<run_id>/
  input/request.json          the model and its settings, fixed for the whole job, plus
                              idle_minutes (default 20) and poll_seconds (default 1)
  inbox/<request>.callback    optional: an https address to call once the request is answered
  inbox/<request>.jsonl.gz    the prompts (Windmill writes each file under a hidden name and
                              renames it, so it is never read half-written)
  outbox/<request>.jsonl.gz   the answers
  outbox/<request>.json       the request's summary, written last: the request is done
  outbox/<request>.error.json instead, when the file wasn't a valid request
  status.json                 what the job is doing now: loading, waiting, answering, stopped
  events.log                  one line per finished request, for whoever is waiting
  output/requests.jsonl       one line per finished request
  output/summary.json         at the end: versions, totals and why it stopped
  stop                        made by Windmill to ask the job to stop
```

A prompt is one of two kinds. A label prompt (`"labels": [...]`, llm_batch's format) is
answered with one label and each label's probability. A generation prompt (`"generate":
{...}`) is answered with written text: it may set `max_tokens`, `temperature`, `top_p`,
`stop`, `seed`, a `json_schema` the answer must match, and `thinking`; it may list `tools`
(OpenAI's format), which the chat template shows the model, and its calls come back in the
text as Qwen's `<tool_call>` blocks. `llm_serve.py`'s docstring has the details.

A request is data only. The job checks every prompt against fixed limits and refuses a file
that isn't valid; nothing in a request can choose the model, a file or a command, and tools
are only descriptions for the model to read (the job never runs them).

Every request in `inbox/` is worked on at once. The job drives vLLM's engine step by step
(continuous batching, as vLLM's own server does): it keeps about `max_num_seqs` prompts
running, takes them from each waiting request in turn, and puts the next prompt in as soon as
one finishes. So a short request that arrives while a long answer is being written is answered
straight away, and many small requests (as the platform's gateway sends) fill the GPU
together. A request's answers are written to `outbox/` when its last prompt finishes.

The job stops when it is asked to (after answering what is already in `inbox/`), when
nothing has arrived for `idle_minutes` (so it doesn't hold a shared GPU doing nothing), or
15 minutes before its time limit, when SLURM sends it `USR1`: it takes nothing new, gives the
prompts already started up to 10 minutes to finish, saves their answers, and leaves the
unfinished requests in `inbox/`. If SLURM requeues it in the same folder, it carries on
where it stopped.

To watch the prompts going in and the answers coming out of a running job, live, run
`~/narratrace/env/bin/python ~/narratrace/slurm/watch_prompts.py` on Blythe in a second
terminal (it reads the newest gateway run folder; give a run folder to watch another; add
`--full` for whole prompts). It only reads files.

Each new Windmill login needs a new 2FA code, so adding prompts from separate Windmill runs
suits batches of work. The platform's LLM gateway instead keeps one login open per person,
which makes one question at a time practical too. Every request leaves two small files in `outbox/`, and
run folders live in the home folder (100,000 files at most), so send prompts in batches.

## On Blythe

Home folders on Blythe are small: 2 GB and 100,000 files (`mmlsquota --block-size auto`).
Project work belongs in your shared directory, which is on SSDs. For Rowan that is
`/springbrook/share/dcsug/csujxb/narratrace` (yours is likely the same with your own
username; check that the folder above it exists). Other users of the share can see it, so it
is kept private (`chmod 700`). The Python environment in it runs under your account, so
nobody else may be able to change it. The big folders live there and are linked
from `~/narratrace`, so the job files and Windmill only ever use `~/narratrace/...`:

```text
~/narratrace/
  slurm/                     a clone of this repository
  jobs -> slurm/jobs         the job files Windmill may submit
  runs/<run_id>/             one folder per run, made by Windmill (small, so kept in home):
    input/                   request.json and prompts.jsonl.gz
    output/                  results.jsonl.gz and summary.json, written by the job
    slurm-<job>.out          the job's log
    callback_url             the one-time Windmill address the job calls when it ends
    inbox/, outbox/, status.json   llm_serve's mailbox (see above)
  env -> share               Python 3.12 with vLLM, made by setup_env.sbatch
  models -> share            the model, at a fixed commit (NARRATRACE_REVISION)
  hf-home, uv-cache, vllm-cache, python, cache -> share
                             Hugging Face, uv and compiler caches, and uv's own Python
```

Setting it up, once per person, logged in to Blythe as yourself:

```sh
git clone https://github.com/NarraTrace-4YP/narratrace-slurm.git ~/narratrace/slurm
ln -s ~/narratrace/slurm/jobs ~/narratrace/jobs

STORE=/springbrook/share/dcsug/$USER/narratrace   # your own share folder
mkdir -p "$STORE" && chmod 700 "$(dirname "$STORE")" "$STORE"
cd ~/narratrace
for d in env models hf-home uv-cache vllm-cache python cache; do
  mkdir -p "$STORE/$d" && ln -sfn "$STORE/$d" "$d"
done
sbatch jobs/setup_env.sbatch       # then check setup-<job>.out
```

`setup_env.sbatch` refuses to start if any of those folders isn't a link, so it can't fill
the home folder by mistake. It downloads vLLM and the model (about 30 GB) into your share.
Then check it from Windmill, logged in as yourself: `f/dhr/tools/llm_server` with action
`start` and a test prompt (see the platform's `f/dhr/README.md`).

To update later: `cd ~/narratrace/slurm && git pull`. Pull only from `main`, so what runs
on Blythe is what was reviewed.

## Developing

```sh
uv sync
uv run pytest            # vLLM is replaced by a stand-in, so no GPU is needed
uv run ruff format . && uv run ruff check .
```

The real model is checked by running the platform's pilot on Blythe.
