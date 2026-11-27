#!/usr/bin/env bash
# 独立普通入口后，保持同一模型实例只切换分片通信/门控。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code=$(cd -- "$here/../../../.." && pwd)
config=$(realpath -- "${1:?需要候选配置}")
root=$(realpath -m -- "${2:?需要新输出目录}")
accepted=$(realpath -- "${3:?需要已验收的qualification目录}")
test ! -e "$root"
mkdir -- "$root"
trap 'printf "%s\n" "$?" > "$root/qualification-exit-code"' EXIT
export IFAB_AFS_HOME="$code/runtime"
export OMP_NUM_THREADS=4 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export H3_AUTOTUNE_USE_ENTRY_DEFAULTS=1
model=${H3_AUTOTUNE_MODEL:-/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3}
cp -- "$config" "$root/candidate.env"
bash "$here/../../adapters/check-h3-official.sh" "$config" "$root/official"
cmp -- "$root/official/video.mp4" "$accepted/official/video.mp4"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$config"
bash "$code/optimization/run-resident-measurement.sh" indexed "$root/steady-paired" \
    --model "$model" --sequence A,A,A,A,A,A,A,A,B,C,A \
    --peer-gate-variants off,on,off,on,on,off,off,on,on,on,on
previous="$accepted/steady-paired/sequence-validation.json"
dog=$(jq -er '[.[] | select(.case == "A")][0].reference' "$previous")
square=$(jq -er '[.[] | select(.case == "B")][0].reference' "$previous")
last=$(jq -er '[.[] | select(.case == "C")][0].reference' "$previous")
python "$code/optimization/validate-sequence.py" "$root/steady-paired" \
    --dog-reference "$dog" --square-reference "$(dirname -- "$square")" \
    --last-frame-reference "$(dirname -- "$last")" > "$root/steady-quality.log" 2>&1
python "$here/summarize-peer-gate.py" "$root" "$accepted" > "$root/summary.log" 2>&1
