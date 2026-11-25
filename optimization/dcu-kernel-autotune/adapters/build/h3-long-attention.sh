#!/usr/bin/env bash
set -euo pipefail
install_dir=${1:?}
if [[ "$install_dir" == */baseline/install ]]; then
  mkdir -- "$install_dir/attention"
  cp -p -- "$H3_ATTENTION_AOT_BUILD"/dense-attention-aot.{so,hsaco} \
    "$H3_ATTENTION_AOT_BUILD/dense-attention-aot-build.json" "$install_dir/attention/"
  sha256sum "$install_dir"/attention/dense-attention-aot.{so,hsaco}
else
  source "$SOURCE_ROOT/flags.conf"
  # 避免同秒、同尺寸的源码增量误读上一轮 bytecode；只清理私有源码缓存。
  python -B - "$SOURCE_ROOT/optimization/adaln" <<'PYCACHE'
from pathlib import Path
import sys
for path in (Path(sys.argv[1]) / "__pycache__").glob("*.pyc"):
    path.unlink()
PYCACHE
  H3_AOT_SAVE_IR=1 PYTHONDONTWRITEBYTECODE=1 timeout --verbose --kill-after=15s "${H3_AUTOTUNE_BUILD_TIMEOUT_SECONDS:-180}s" \
    python -u "$SOURCE_ROOT/optimization/adaln/build-attention-aot.py" \
    "$install_dir/attention" --fp32-scale --aligned-strides --query-block "${AOT_QUERY_BLOCK:-32}" --key-block "${AOT_KEY_BLOCK:-32}" \
    --warps "${AOT_WARPS:-2}" --stages "${AOT_STAGES:-1}" --kpack "${AOT_KPACK:-2}" \
    --matrix-instr-nonkdim "${AOT_MATRIX_INSTR_NONKDIM:-0}" \
    --waves-per-eu "$WAVES_PER_EU" --schedule-hint "$SCHEDULE_HINT"
fi
