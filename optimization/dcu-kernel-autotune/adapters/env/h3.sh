#!/usr/bin/env bash
export H3_AUTOTUNE_PROJECT
H3_AUTOTUNE_PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
export H3_AUTOTUNE_CODE_ROOT="$H3_AUTOTUNE_PROJECT/submission-xsh"
if [[ ! -f "$H3_AUTOTUNE_CODE_ROOT/optimization/bench-attention.py" ]]; then
  export H3_AUTOTUNE_CODE_ROOT="$H3_AUTOTUNE_PROJECT"
fi
export IFAB_AFS_HOME="$H3_AUTOTUNE_PROJECT/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$H3_AUTOTUNE_CODE_ROOT/optimization/load-candidate-config.sh" \
  "${H3_AUTOTUNE_BASE_CONFIG:-$H3_AUTOTUNE_PROJECT/optimization/selected-best.env}"
export HIP_VISIBLE_DEVICES="${H3_AUTOTUNE_DEVICE:-0}"
