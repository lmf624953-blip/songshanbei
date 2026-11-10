#!/usr/bin/env bash
set -eo pipefail
cd "${IFAB_AFS_HOME:?}"
source scripts/activate.sh
export PATH="$IFAB_AFS_HOME/.venv/bin:$PATH"
export SONGSHAN_STORAGE="$IFAB_AFS_HOME/h3-runtime"
export PYTHONPATH="$IFAB_AFS_HOME:${PYTHONPATH:-}"
export TRITON_CACHE_MANAGER='songshan_h3.cache_compat:SameDirectoryCacheManager'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
H3_RUNTIME_CACHE=$(python -c 'import tempfile;print(tempfile.mkdtemp(prefix="h3-inference-cache-"))')
export TRITON_CACHE_DIR="$H3_RUNTIME_CACHE/triton" NINETOOTHED_CACHE_DIR="$H3_RUNTIME_CACHE/ninetoothed"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
python -u scripts/run_official_direct.py
