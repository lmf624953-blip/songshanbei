#!/usr/bin/env bash
# 与无profiler计时分开执行；保留hipprof原始TXT和完整八卡时间线。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code=$(cd -- "$here/../../../.." && pwd)
config=$(realpath -- "${1:?需要候选配置}")
out=$(realpath -m -- "${2:?需要新输出目录}")
test ! -e "$out"
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/exit-code"' EXIT
export IFAB_AFS_HOME="$code/runtime"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$config"
test -r "$H3_PEER_GATE_BUILD/peer-runtime-build.json"
command=(python "$here/bench-peer-reduce.py" "$INFINI_ROOT/lib/libinfiniccl.so"
    "$H3_PEER_GATE_BUILD/peer-gate-bridge.so" OUTPUT
    --bridge-build "$H3_PEER_GATE_BUILD" --indexed-build "$H3_INDEXED_BUILD"
    --gather write --vectors 1 --blocks 5120 --pairs 1 --counts 148216320)
command[4]="$out/pmc-result.json"
hipprof --pmc --kernel-name h3_peer_gate_kernel --devices 0,1,2,3,4,5,6,7 \
    -o "$out/pmc" "${command[@]}" > "$out/pmc.log" 2>&1
test -s "$out/pmc.txt"
command[4]="$out/trace-result.json"
hipprof --hip-trace --rccl-trace --barrier --group-stream --output-type 0 \
    --devices 0,1,2,3,4,5,6,7 -o "$out/trace" "${command[@]}" > "$out/trace.log" 2>&1
test -s "$out/trace_stream.json"
