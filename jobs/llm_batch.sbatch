#!/bin/bash
#SBATCH --job-name=narratrace-llm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --exclude=sea004
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Answers a batch of multiple-choice prompts with an LLM (vLLM, one L40 GPU).
# sea004 is left out: on 20 Nov 2026 two jobs failed there while starting the model, where
# every other L40 node had worked (remove the --exclude line once it is known to be fixed).
# Windmill runs it from a run folder, sometimes with a shorter or longer time limit:
#   sbatch --chdir ~/narratrace/runs/<run> [--time=HH:MM:SS] ~/narratrace/jobs/llm_batch.sbatch
# Reads input/request.json and input/prompts.jsonl.gz; writes output/results.jsonl.gz and
# output/summary.json (see ~/narratrace/slurm/llm_batch.py).
# When it ends, even after an error, it wakes the Windmill run using the address in
# callback_url (only there for runs started from Windmill).

set -uo pipefail
NARRATRACE="$HOME/narratrace"

tell_windmill() {
  local code=$?
  [ -f callback_url ] || return 0
  for attempt in 1 2 3 4 5 6; do
    if curl -fsS -m 20 -o /dev/null -X POST -H 'Content-Type: application/json' \
        -d "{\"slurm_job_id\": \"$SLURM_JOB_ID\", \"exit_code\": $code}" "$(cat callback_url)"; then
      echo "Woke the Windmill run (attempt $attempt)"
      return 0
    fi
    sleep 20
  done
  echo "Couldn't reach Windmill; it will notice at its next check"
}
trap tell_windmill EXIT
trap 'exit 143' TERM

echo "Job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Models are read from ~/narratrace/models only; nothing is downloaded during a run.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$NARRATRACE/hf-home" VLLM_CACHE_ROOT="$NARRATRACE/vllm-cache"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# Compiler caches (Triton, FlashInfer, ...) go to the project share, not the small home folder.
export XDG_CACHE_HOME="$NARRATRACE/cache" TRITON_HOME="$NARRATRACE/cache"
export FLASHINFER_WORKSPACE_BASE="$NARRATRACE/cache"

# Which version of this repository answered, stored in output/summary.json ("-dirty" if
# the clone on Blythe has changes that aren't committed).
CODE="$NARRATRACE/slurm"
NARRATRACE_CODE_COMMIT="$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)"
[ -n "$(git -C "$CODE" status --porcelain 2>/dev/null)" ] && NARRATRACE_CODE_COMMIT+="-dirty"
export NARRATRACE_CODE_COMMIT

"$NARRATRACE/env/bin/python" "$NARRATRACE/slurm/llm_batch.py" \
  --models "$NARRATRACE/models" input output
