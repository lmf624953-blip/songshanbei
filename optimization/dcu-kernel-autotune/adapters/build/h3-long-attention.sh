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
  H3_AOT_SAVE_IR=1 python -u "$SOURCE_ROOT/optimization/adaln/build-attention-aot.py" \
    "$install_dir/attention" --fp32-scale --aligned-strides --query-block "${AOT_QUERY_BLOCK:-32}" --key-block 32 \
    --warps "${AOT_WARPS:-2}" --stages "${AOT_STAGES:-1}" --kpack 2 --waves-per-eu "$WAVES_PER_EU" --schedule-hint "$SCHEDULE_HINT"
fi
