#!/bin/bash
#SBATCH --job-name=narratrace-serve
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --exclude=sea004
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=slurm-%j.out

# Keeps a model loaded on one L40 (or split over more, when Windmill asks for them with
# sbatch --gres, as the LLM gateway does for full-precision Qwen3.8-27B on two) and answers
# prompts as Windmill adds them to the run folder's inbox/ (llm_serve.py describes the
# files). It stops when asked (a "stop" file), after idle_minutes with nothing to do, or near its time limit: 15 minutes before it,
# SLURM sends this script USR1, which is passed on so the server finishes what it has started.
# Windmill runs it from a run folder, usually with a shorter time limit:
#   sbatch --chdir ~/narratrace/runs/<run> [--time=HH:MM:SS] ~/narratrace/jobs/llm_serve.sbatch
# Reads input/request.json; Windmill adds inbox/<request>.jsonl.gz files while it runs.
# When it ends, even after an error, it wakes the Windmill run using the address in
# callback_url (only there if Windmill left one). Each request has its own callback too.

set -uo pipefail
NARRATRACE="$HOME/narratrace"

job_tmp=""  # node-local temporary folder, only made for a job with more than one GPU
tell_windmill() {
  local code=$?
  [ -n "$job_tmp" ] && rm -rf -- "$job_tmp"
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

echo "Job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# As llm_batch.sbatch: models only from ~/narratrace/models, caches in the project share.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$NARRATRACE/hf-home" VLLM_CACHE_ROOT="$NARRATRACE/vllm-cache"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export XDG_CACHE_HOME="$NARRATRACE/cache" TRITON_HOME="$NARRATRACE/cache"
export FLASHINFER_WORKSPACE_BASE="$NARRATRACE/cache"

# More than one GPU: the model is split over them (tensor parallel), so they must talk to each
# other. These are the settings of test job 1389794 (Rowan, 25 Nov 2026, sea003), the first
# two-GPU run that loaded and answered; earlier ones hung while the GPUs set up their links.
# - Blythe's two L40s in a job can sit on different CPU sockets (nvidia-smi topo: SYS), so
#   direct GPU-to-GPU copies are switched off; NCCL goes through shared memory instead.
# - Everything is on one node, so the processes meet on the loopback address.
# - vLLM's start-up file and the compiled-kernel caches go to the node's own disk, not GPFS.
# Then a 20-second check (check_gpus.py) runs before the model's several-minute load, so a job
# whose GPUs can't talk fails at once, with the reason in status.json for the gateway.
gpus="${SLURM_GPUS_ON_NODE:-}"
[ -n "$gpus" ] || gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
if [ "${gpus:-1}" -gt 1 ]; then
  local_base=/tmp
  if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "$SLURM_TMPDIR" ] && [ -w "$SLURM_TMPDIR" ]; then
    local_base="$SLURM_TMPDIR"
  fi
  job_tmp="$(mktemp -d "$local_base/narratrace-serve-${SLURM_JOB_ID:-x}.XXXXXXXX")"
  mkdir -p "$job_tmp"/{tmp,xdg,triton,torchinductor,vllm,cuda}
  export TMPDIR="$job_tmp/tmp" XDG_CACHE_HOME="$job_tmp/xdg" TRITON_CACHE_DIR="$job_tmp/triton"
  export TORCHINDUCTOR_CACHE_DIR="$job_tmp/torchinductor" VLLM_CACHE_ROOT="$job_tmp/vllm"
  export CUDA_CACHE_PATH="$job_tmp/cuda"
  export NCCL_P2P_DISABLE=1 GLOO_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=4
  unset NCCL_SOCKET_IFNAME
  echo "$gpus GPUs: checking they can talk to each other (temporary files in $job_tmp)"
  nvidia-smi topo -m || true
  if ! timeout --signal=TERM --kill-after=10s 180s \
      "$NARRATRACE/env/bin/python" -m torch.distributed.run --standalone --nnodes=1 \
      --nproc-per-node="$gpus" "$NARRATRACE/slurm/check_gpus.py"; then
    reason="the $gpus GPUs on $(hostname) couldn't talk to each other (check_gpus.py failed or took over 3 minutes)"
    echo "ERROR $reason"
    printf '{"state": "failed", "error": "%s", "slurm_job_id": "%s", "updated": "%s"}\n' \
      "$reason" "$SLURM_JOB_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > status.json
    exit 1
  fi
fi

CODE="$NARRATRACE/slurm"
NARRATRACE_CODE_COMMIT="$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)"
[ -n "$(git -C "$CODE" status --porcelain 2>/dev/null)" ] && NARRATRACE_CODE_COMMIT+="-dirty"
export NARRATRACE_CODE_COMMIT

# The server runs in the background so this script can pass SLURM's signals on to it:
# bash only runs a trap between commands, and waiting on a background process lets it.
"$NARRATRACE/env/bin/python" "$NARRATRACE/slurm/llm_serve.py" --models "$NARRATRACE/models" . &
server=$!
trap 'echo "Time limit near: telling the server to stop"; kill -USR1 "$server" 2>/dev/null' USR1
trap 'kill -TERM "$server" 2>/dev/null; exit 143' TERM
status=0
while true; do
  wait "$server"
  status=$?
  # wait also returns when a trapped signal arrives; carry on until the server has ended.
  kill -0 "$server" 2>/dev/null || break
done
exit "$status"
