#!/usr/bin/env bash
# 加载选定优化实现，原样转交官方入口的 task 和所有输入参数。
set -eo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config="$project/optimization/selected-best.env"
if [[ "${1:-}" == --config ]]; then
  config=$(realpath -- "${2:?--config requires a file}")
  shift 2
fi
if [[ $# == 0 || "${1:-}" == --help ]]; then
  printf 'Usage: bash optimization/run-optimized.sh [--config FILE] fl2va [official arguments]\n'
  printf 'Example: --model-path=PATH --first-frame=PATH --text=TEXT --gpus=0,1,2,3,4,5,6,7 --steps=50 --num-frames=124 --seed=42 --output=PATH\n'
  exit 0
fi
test -s "$config" || { printf 'Selected configuration is missing: %s\n' "$config" >&2; exit 2; }
runtime="$project/runtime"
export IFAB_AFS_HOME="$runtime"
source "$runtime/scripts/unified_dog_env.sh"
source "$project/optimization/load-candidate-config.sh" "$config"
if [[ -n "${H3_BENCH_JIT_CACHE:-}" ]]; then
  export TRITON_CACHE_DIR="$H3_BENCH_JIT_CACHE/triton"
  export NINETOOTHED_CACHE_DIR="$H3_BENCH_JIT_CACHE/ninetoothed"
fi
overlay=$(cat "${H3_ADALN_BUILD:?Selected native build required}/native-overlay.txt")
export PYTHONPATH="$overlay:$PYTHONPATH"
printf 'H3 selected configuration: %s\n' "$config"
exec "$runtime/.venv/bin/python" -u "$runtime/scripts/launch_h3.py" \
  "$project/optimization/adaln/launch.py" "$@"
