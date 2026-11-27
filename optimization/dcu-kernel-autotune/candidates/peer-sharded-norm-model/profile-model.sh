#!/usr/bin/env bash
# 正式无profiler计时结束后单独采集；trace来自完整模型，PMC来自相同模型桥镜像的独立八卡检查。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code=$(cd -- "$here/../../../.." && pwd)
mode=${1:?需要trace或pmc-gate或pmc-norm}
config=$(realpath -- "${2:?需要候选配置}")
out=$(realpath -m -- "${3:?需要新输出目录}")
test ! -e "$out"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$config"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
python - "$here" <<'PYCHECK'
import os,sys
sys.path.insert(0,sys.argv[1])
from model_identity import check
check(os.environ['H3_PEER_SHARDED_NORM_BUILD'],os.environ['H3_ADALN_BUILD'],os.environ['H3_PEER_GATE_BUILD'],os.environ['H3_WIDE_NORM_INDEXED_BUILD'])
PYCHECK
case "$mode" in trace|pmc-gate|pmc-norm) ;; *) printf '未知采集模式：%s\n' "$mode" >&2; exit 2 ;; esac
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/exit-code"' EXIT
cp -- "$config" "$out/candidate.env"
if [[ "$mode" == trace ]]; then
    model=${H3_AUTOTUNE_MODEL:-/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3}
    captures=${H3_AUTOTUNE_CAPTURES:-/public/home/acvduipmem/xsh/songshanbei/optimization/kernel-baseline/denoise-captures}
    command=(hipprof --hip-trace --rccl-trace --barrier --devices 0,1,2,3,4,5,6,7
        --group-stream --flush-interval 100 --buffer-size 10000 --trace-off --output-type 0 -o "$out/hipprof"
        bash "$code/optimization/run-optimized.sh" --config "$config" --replay-run
        --model "$model" --captures "$captures" --out "$out/replay" --steps 25
        --candidate-only --profile-once --profile-drain-seconds 2)
else
    kernel=h3_peer_gate_kernel
    [[ "$mode" != pmc-norm ]] || kernel=wide_norm_indexed_kernel
    command=(hipprof --pmc --kernel-name "$kernel" --devices 0,1,2,3,4,5,6,7 -o "$out/pmc"
        python "$here/check-model.py" "$H3_PEER_SHARDED_NORM_BUILD" "$out/result.json"
        --reference "$H3_PEER_GATE_BUILD" --indexed "$H3_INDEXED_BUILD" --norm "$H3_WIDE_NORM_INDEXED_BUILD" --rows 27570)
fi
printf '%q ' "${command[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
"${command[@]}" > "$out/run.log" 2>&1
if [[ "$mode" == trace ]]; then
    test -s "$out/hipprof_stream.json"
    test -s "$out/replay/results.json"
else
    test -s "$out/pmc.txt"
fi
