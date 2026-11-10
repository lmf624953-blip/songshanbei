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
python -u - <<'PY'
import torch,infinicore,infinilm,importlib.metadata as md
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder,MiniMaxH3TransformerRunner
assert torch.cuda.device_count()==8,torch.cuda.device_count()
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx936')
print('NATIVE_RUNTIME_IMPORTS_PASS',md.version('InfiniCore'),md.version('InfiniLM'),flush=True)
PY
python -u scripts/check_attention_bridge.py > artifacts/deployed-attention-devices.log 2>&1
python -u scripts/run_public_fl2va.py > artifacts/deployed-public-fl2va.log 2>&1
echo DEPLOYED_FL2VA_GENERATION_EXIT=0
