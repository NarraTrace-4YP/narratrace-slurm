# NarraTrace SLURM jobs

The fixed jobs NarraTrace runs on Blythe, the University of Warwick's GPU cluster (SCRTP).
The platform (`NarraTrace-4YP/4yp_platform`) submits them from Windmill. The first user is
O1 stage 5, `f/dhr/O1/choose_template_section`, which runs an LLM to map each review onto
its statutory template.

This repository is kept apart from the platform because it is what runs on Blythe, under
Rowan's university account. Its clone on Blythe changes only when Rowan pulls it there. The
platform's Windmill code changes separately.

## How Windmill uses it

Windmill logs in over SSH with Rowan's key and a 2FA code made from his 2FA secret. SCRTP
allowed this on 19 Nov 2026, until they offer a web interface. Only Rowan's Windmill runs can
do this (`check_owner()` in the platform's `f/dhr/blythe.py`). Windmill can only:

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

Each job's last step POSTs to the address in its run folder's `callback_url` (up to six
tries, 20 s apart). This wakes the waiting Windmill run. Windmill then asks SLURM whether
the job has really ended, so a lost or early call does no harm.

`llm_batch` records this repository's commit in `output/summary.json` (`code_commit`,
ending in `-dirty` if the clone has uncommitted changes). The platform stores that file with
the answers, so every result can be traced to the code that produced it.

## On Blythe

```text
~/narratrace/
  slurm/                     a clone of this repository
  jobs -> slurm/jobs         the job files Windmill may submit
  env/                       Python 3.12 with vLLM, made by setup_env.sbatch
  models/<org>/<name>/       the model, at a fixed commit (NARRATRACE_REVISION)
  runs/<run_id>/             one folder per run, made by Windmill:
    input/                   request.json and prompts.jsonl.gz
    output/                  results.jsonl.gz and summary.json, written by the job
    slurm-<job>.out          the job's log
    callback_url             the one-time Windmill address the job calls when it ends
```

Setting it up (Rowan):

1. `git clone https://github.com/NarraTrace-4YP/narratrace-slurm.git ~/narratrace/slurm`
2. Point the jobs folder at the clone:
   `mv ~/narratrace/jobs ~/narratrace/jobs.old && ln -s ~/narratrace/slurm/jobs ~/narratrace/jobs`
3. `cd ~/narratrace && sbatch jobs/setup_env.sbatch`, and check `setup-<job>.out` when it
   finishes.
4. From Windmill, run `u/rshields2004/blythe_run_job` with `test_gpu`. The driver it reports
   must be 580 or newer for vLLM 0.34's CUDA 13.

To update later: `cd ~/narratrace/slurm && git pull`. Pull only from `main`, so what runs
on Blythe is what was reviewed.

## Developing

```sh
uv sync
uv run pytest            # vLLM is replaced by a stand-in, so no GPU is needed
uv run ruff format . && uv run ruff check .
```

The real model is checked by running the platform's pilot on Blythe.
