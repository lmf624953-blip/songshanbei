#!/usr/bin/env bash
# Run in a normal network-enabled SSH terminal. Installs into project .venv only.
# This prepares dependencies, not a compiled InfiniLM or a competition submission.
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/activate.sh"
source "$project_root/env/storage.sh"
set -u
mkdir -p "$project_root/logs" "$project_root/artifacts" "$project_root/downloads"
exec > >(tee -a "$project_root/logs/prepare_dependencies.log") 2>&1
trap 'echo "Failed at line $LINENO; see logs/prepare_dependencies.log" >&2' ERR

bash "$project_root/scripts/fetch_sources.sh"

# Source builds support this host's glibc 2.28 and replace an unusable bundle.
bash "$project_root/scripts/setup_xmake.sh"
hash -r
xmake --version

# Pin the inspected H3 source rather than following a moving main branch.
diffusers_dir="$SONGSHAN_STORAGE/sources/diffusers"
diffusers_revision="86ef420b17363c28e36ebad6b58555e9aa8feda2"
if [[ ! -e "$diffusers_dir" ]]; then
    git clone --depth 1 https://github.com/huggingface/diffusers.git "$diffusers_dir"
    git -C "$diffusers_dir" fetch --depth 1 origin "$diffusers_revision"
    git -C "$diffusers_dir" checkout --detach "$diffusers_revision"
fi
if [[ "$(git -C "$diffusers_dir" rev-parse HEAD)" != "$diffusers_revision" ]]; then
    echo "Diffusers revision differs from the inspected pin: $diffusers_revision" >&2
    echo "Existing checkout retained; inspect it before changing revisions: $diffusers_dir" >&2
    exit 1
fi
git -C "$diffusers_dir" rev-parse HEAD > "$project_root/artifacts/diffusers.commit"
if [[ ! -d "$diffusers_dir/src/diffusers/modular_pipelines/minimax_h3" ]]; then
    echo "Downloaded Diffusers does not contain MiniMax H3; inspect its revision." >&2
    exit 1
fi

python - <<'PY'
import torch
from importlib.metadata import version
assert torch.version.__hcu_version__ == '2.9.0+das.opt1.dtk2604'
assert version('torch') == '2.9.0+das.opt1.dtk2604'
assert version('triton') == '3.5.1+das.opt1.dtk2604.torch290'
PY

# Resolve the entire dependency set together, before changing installed packages.
# In particular, Hub 1.x and Transformers 4.x cannot be installed together.
install_args=(
    -c "$project_root/env/inference-constraints.txt"
    hatchling pybind11 accelerate transformers huggingface-hub
    safetensors av imageio imageio-ffmpeg
    scipy scikit-image einops sentencepiece ftfy janus xxhash
    -e "$SONGSHAN_STORAGE/sources/ninetoothed"
    -e "$SONGSHAN_STORAGE/sources/ntops"
    -e "$diffusers_dir"
)
python -m pip install --dry-run --report "$project_root/artifacts/dependency-plan.json" \
    "${install_args[@]}"
python -m pip freeze > "$project_root/artifacts/dependencies-before-install.txt"
python -m pip install --report "$project_root/artifacts/dependency-install.json" \
    "${install_args[@]}"
bash "$project_root/scripts/prepare_vision_dependencies.sh" --install-only
python "$project_root/scripts/check_inference_dependencies.py"
python -m pip freeze > "$project_root/artifacts/inference-dependencies.freeze.txt"
echo "Dependency preparation passed. InfiniCore/InfiniLM still require compilation."
if [[ "${1:-}" == "--with-model" ]]; then
    python "$project_root/scripts/download_model.py"
fi
