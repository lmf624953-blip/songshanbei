#!/usr/bin/env bash
# 从保留 LLVM IR 的 Q128 packed 构建重建最新长注意力候选。
set -eo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=$(cd -- "$scripts/../../../.." && pwd)
control=$(realpath -- "${1:?需要保留 LLVM IR 的 Q128 packed 注意力目录}")
config=$(realpath -- "${2:?需要已有的推理配置文件}")
out=$(realpath -m -- "${3:?需要尚不存在的候选输出目录}")
test -f "$control/dense-attention-packed.llir"
test -f "$config"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"

mkdir -- "$out"
python "$scripts/build-softmax-dpp-arithmetic.py" "$control" "$out/four-row-dpp" \
    all --xor4 both --bundle both --bundle-width 4
python "$scripts/build-mmac-issue.py" "$out/four-row-dpp" "$out/joint-kv" \
    --joint-kv-wait 4 --elide-v pack --kv-zero miss --prefetch-v direct \
    --qk-wait-width 2 --pv-wait-width 4 --pack-pv-rescale --pack-qk-scale
python "$scripts/build-k-cross-block.py" "$out/joint-kv" "$out/k-induction" \
    after-pv --pair-with-lds --address-induction k
python "$scripts/build-softmax-schedule.py" "$out/k-induction" "$out/folded" --fold-max only
python "$scripts/build-softmax-schedule.py" "$out/folded" "$out/attention" --row-sum interleave

cp -- "$config" "$out/candidate.env"
printf '\nexport H3_ATTENTION_AOT_BUILD=%q\n' "$out/attention" >> "$out/candidate.env"
printf '%s\n' "$out/candidate.env"
