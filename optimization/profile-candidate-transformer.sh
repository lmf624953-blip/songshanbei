#!/usr/bin/env bash
# 只采集一个已预热真实 Transformer 调用，正式常驻测量另行执行。
set -eo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config=$(realpath -- "${1:?Provide the candidate config}")
model=$(realpath -- "${2:?Provide the model weights}")
captures=$(realpath -- "${3:?Provide the diagnostic input captures}")
out=$(realpath -m -- "${4:?Provide a new output directory}")
test ! -e "$out"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$project/optimization/load-candidate-config.sh" "$config"
if [[ -n "${H3_BENCH_JIT_CACHE:-}" ]]; then
  export TRITON_CACHE_DIR="$H3_BENCH_JIT_CACHE/triton"
  export NINETOOTHED_CACHE_DIR="$H3_BENCH_JIT_CACHE/ninetoothed"
fi
overlay=$(cat "$H3_ADALN_BUILD/native-overlay.txt")
status=0
PYTHONPATH="$overlay:$PYTHONPATH" hipprof --hip-trace --rccl-trace --devices 0 \
    --trace-off --output-type 0 --segment-size 1000000 -o "$out-hipprof" \
    python -u "$IFAB_AFS_HOME/scripts/launch_h3.py" "$project/optimization/adaln/launch.py" \
    --replay-run --model "$model" --captures "$captures" \
    --out "$out" --steps 25 --profile-once > "$out.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out.exit-code"
test "$status" = 0
python "$project/optimization/summarize-profile.py" "$out.log" "$out-hipprof.json" \
    --output "$out/summary.json"
