#!/bin/bash
#SBATCH --job-name=narratrace-setup
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=04:00:00
#SBATCH --output=setup-%j.out

# Sets up ~/narratrace on Blythe for the LLM jobs, once (and again after changing a version
# below). Needs no GPU, so it runs on the compute partition:
#   cd ~/narratrace && sbatch jobs/setup_env.sbatch [model]
# It also works straight from the login node: bash jobs/setup_env.sbatch [model]
#
#   ~/narratrace/env/      Python 3.12 with vLLM (Blythe's own Python 3.9 is too old)
#   ~/narratrace/models/   the model, e.g. models/Qwen/Qwen3.8-27B-FP8, downloaded at a fixed
#                          Hugging Face commit, which is saved in NARRATRACE_REVISION there
#   env-freeze.txt         every package version installed
# Safe to run again: what is already there is kept.

set -euo pipefail
NARRATRACE="$HOME/narratrace"
UV_VERSION=0.11.32
VLLM_VERSION=0.34.0
MODEL="${1:-Qwen/Qwen3.8-27B-FP8}"

export HF_HOME="$NARRATRACE/hf-home"
export UV_CACHE_DIR="$NARRATRACE/uv-cache" UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

echo "Setting up $NARRATRACE on $(hostname), $(date -Is)"
echo "Space in your home folder:"
df -h "$HOME" | tail -n 1
quota -s 2>/dev/null || true

if ! uv --version 2>/dev/null | grep -q "^uv $UV_VERSION"; then
  echo "Installing uv $UV_VERSION into ~/.local/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
    | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

if [ ! -x "$NARRATRACE/env/bin/python" ]; then
  echo "Making the Python 3.12 environment"
  uv venv --python 3.12 "$NARRATRACE/env"
fi
echo "Installing vLLM $VLLM_VERSION (several GB; takes a while)"
uv pip install --python "$NARRATRACE/env/bin/python" "vllm==$VLLM_VERSION" "huggingface-hub[hf_xet]"
uv pip freeze --python "$NARRATRACE/env/bin/python" > "$NARRATRACE/env-freeze.txt"
"$NARRATRACE/env/bin/python" -c \
  "import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__, '| built for CUDA', torch.version.cuda)"

echo "Downloading $MODEL"
"$NARRATRACE/env/bin/python" - "$MODEL" "$NARRATRACE/models" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

model, root = sys.argv[1], Path(sys.argv[2])
target = root / model
saved = target / "NARRATRACE_REVISION"
revision = saved.read_text().strip() if saved.exists() else HfApi().model_info(model).sha
snapshot_download(model, revision=revision, local_dir=target)
saved.write_text(revision + "\n")
size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e9
print(f"{model} at commit {revision}: {size:.1f} GB in {target}")
PY

uv cache clean >/dev/null 2>&1 || true
echo "Done, $(date -Is). Space now:"
df -h "$HOME" | tail -n 1
