#!/usr/bin/env bash
set -eo pipefail
cd "$IFAB_AFS_HOME"
source scripts/activate.sh
export PATH="$IFAB_AFS_HOME/.venv/bin:$PATH"
export PYTHONPATH="$IFAB_AFS_HOME/.reference-runtime/python:$IFAB_AFS_HOME/.reference-runtime/flash-attn:$IFAB_AFS_HOME"
export LD_LIBRARY_PATH="$IFAB_AFS_HOME/.reference-runtime/python/torch/lib:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -u -c 'import torch; assert torch.cuda.device_count()==8; q=torch.randn(1,4,256,64,device="cuda",dtype=torch.bfloat16); o=torch.nn.functional.scaled_dot_product_attention(q,q,q); torch.cuda.synchronize(); assert torch.isfinite(o).all(); print("REFERENCE_SDPA_PASS",flush=True)'
python -u scripts/run_reference_comparison.py
