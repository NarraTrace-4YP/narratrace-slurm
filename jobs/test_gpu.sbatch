#!/bin/bash
#SBATCH --job-name=narratrace-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

# NarraTrace test job: checks a GPU job runs and Windmill gets its result back.
#   sbatch --chdir ~/narratrace/runs/<run> ~/narratrace/jobs/test_gpu.sbatch
# Reads input/message.txt, writes output/result.json.
# When it ends, even after an error, it wakes the Windmill run using the address in
# callback_url (only there for runs started from Windmill).

set -uo pipefail

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

mkdir -p output
echo "Job $SLURM_JOB_ID on $(hostname), started $(date -Is)"

gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
message=$(cat input/message.txt 2>/dev/null || echo "no input")
python=$(python3 --version 2>&1 || echo "none")
apptainer=$(command -v apptainer || command -v singularity || echo "none")
if curl -sS -o /dev/null --max-time 10 https://huggingface.co; then internet=yes; else internet=no; fi

cat > output/result.json <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "node": "$(hostname)",
  "gpu": "${gpu:-none}",
  "driver": "${driver:-unknown}",
  "message": "$message",
  "python": "$python",
  "apptainer": "$apptainer",
  "internet": "$internet",
  "finished": "$(date -Is)"
}
EOF
echo "Wrote output/result.json"
