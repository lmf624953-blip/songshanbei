#!/usr/bin/env bash
# 依次验证普通官方入口与同一八卡实例的端到端配对。
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
framework=$(cd -- "$script_dir/../.." && pwd)
root=$(realpath -- "${1:?需要实验目录}")
code=$(realpath -- "${2:?需要交付仓库路径}")
base_config=$(realpath -- "${3:?需要对照配置路径}")
accepted=$(realpath -- "${4:?需要已验收的实验目录}")
build=$(realpath -- "${5:?需要候选镜像目录}")
runtime=$(realpath -- "$code/runtime")
model=${H3_AUTOTUNE_MODEL:-/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3}
export IFAB_AFS_HOME="$runtime"
export OMP_NUM_THREADS=4 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export H3_AUTOTUNE_USE_ENTRY_DEFAULTS=1
for path in official steady-paired candidate.env qualification-exit-code; do
    if [[ -e "$root/$path" ]]; then
        printf '已有验收产物 %s\n' "$root/$path" >&2
        exit 2
    fi
done
trap 'printf "%s\n" "$?" > "$root/qualification-exit-code"' EXIT

# 普通入口从候选配置加载镜像，同进程配对从对照配置启动。
cp -- "$base_config" "$root/candidate.env"
printf 'export H3_ATTENTION_AOT_BUILD=%q\n' "$build" >> "$root/candidate.env"
bash "$framework/adapters/check-h3-official.sh" "$root/candidate.env" "$root/official"
cmp -- "$root/official/video.mp4" "$accepted/official/video.mp4"
source "$runtime/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$base_config"
bash "$code/optimization/run-sequence.sh" indexed --model "$model" --out "$root/steady-paired" \
    --sequence A,A,A,A,A,A,A,A,B,C,A --fast-attention-build "$build" \
    --attention-variants base,fast,base,fast,fast,base,base,fast,fast,fast,fast \
    > "$root/steady-paired.log" 2>&1

previous="$accepted/steady-paired/sequence-validation.json"
dog=$(jq -er '[.[] | select(.case == "A")][0].reference' "$previous")
square_video=$(jq -er '[.[] | select(.case == "B")][0].reference' "$previous")
last_video=$(jq -er '[.[] | select(.case == "C")][0].reference' "$previous")
python "$code/optimization/validate-sequence.py" "$root/steady-paired" \
    --dog-reference "$dog" --square-reference "$(dirname -- "$square_video")" \
    --last-frame-reference "$(dirname -- "$last_video")" \
    > "$root/steady-quality.log" 2>&1
printf "0\n" > "$root/qualification-exit-code"
python "$script_dir/summarize-softmax-dpp-steady.py" "$root" "$accepted/steady-paired" "$accepted/official"
