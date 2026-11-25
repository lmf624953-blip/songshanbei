#!/usr/bin/env bash
set -euo pipefail
code="$H3_AUTOTUNE_CODE_ROOT/optimization"
adapter=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "$PROFILER" ]]; then
  read -r -a profile <<< "$PROFILER"
  if [[ " $PROFILER " == *" --pmc "* ]]; then
    profile+=(--pmc-type 3 --kernel-name dense_attention_aot_kernel)
  fi
  "${profile[@]}" python -u "$code/profile-attention-pmc.py" "$INSTALL_DIR/attention" \
    --sequence 27570 22016 --iterations 2 > "$LOG_FILE" 2>&1
else
  baseline="$adapter/../rounds/h3-long-attention/baseline/install/attention"
  PYTHONPATH="$code:$PYTHONPATH" python -u "$adapter/bench-h3-attention.py" \
    "$INSTALL_DIR/attention" "$baseline" --iterations "$BENCHMARK_STEPS" > "$LOG_FILE" 2>&1
fi
