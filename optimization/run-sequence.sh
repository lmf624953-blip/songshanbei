#!/usr/bin/env bash
set -eo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime="$project/runtime"
export IFAB_AFS_HOME="$runtime"
source "$runtime/scripts/unified_dog_env.sh"
if [[ -n "${H3_BENCH_JIT_CACHE:-}" ]]; then
  export TRITON_CACHE_DIR="$H3_BENCH_JIT_CACHE/triton"
  export NINETOOTHED_CACHE_DIR="$H3_BENCH_JIT_CACHE/ninetoothed"
fi
mode=${1:?infinilm, adaln-aot, indexed or reference}
shift
case "$mode" in
  infinilm)
    exec "$runtime/.venv/bin/python" -u "$runtime/scripts/launch_h3.py" \
      "$project/optimization/run-sequence.py" "$@"
    ;;
  adaln-aot|indexed)
    export H3_ADALN_BUILD="${H3_ADALN_BUILD:?Candidate InfiniLM build required}"
    export H3_ADALN_AOT_BUILD="${H3_ADALN_AOT_BUILD:?AdaLN device build required}"
    if [[ "$mode" == indexed ]]; then
      export H3_INDEXED_BUILD="${H3_INDEXED_BUILD:?Indexed device build required}"
    else
      unset H3_INDEXED_BUILD
    fi
    overlay=$(cat "$H3_ADALN_BUILD/native-overlay.txt")
    export PYTHONPATH="$overlay:$PYTHONPATH"
    exec "$runtime/.venv/bin/python" -u "$runtime/scripts/launch_h3.py" \
      "$project/optimization/adaln/launch.py" --sequence-run "$@"
    ;;
  reference)
    export PYTHONPATH="$runtime/.reference-runtime/python:$runtime/.reference-runtime/flash-attn:$PYTHONPATH"
    export LD_LIBRARY_PATH="$runtime/.reference-runtime/python/torch/lib:$LD_LIBRARY_PATH"
    exec "$runtime/.venv/bin/python" -u "$project/optimization/run-sequence.py" --legacy "$@"
    ;;
  *)
    printf 'Unknown backend: %s\n' "$mode" >&2
    exit 2
    ;;
esac
