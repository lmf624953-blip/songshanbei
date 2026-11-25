#!/usr/bin/env bash
# 在相同八卡模型实例中回放真实早、中、晚期输入，保留默认 v3 对照。
set -eo pipefail
adapter=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
framework=$(cd -- "$adapter/.." && pwd)
project=$(cd -- "$framework/../.." && pwd)
round=${1:?需要轮次编号，例如 round_011}
[[ "$round" =~ ^round_[0-9]{3}$ ]] || exit 2
candidate="$framework/rounds/h3-long-attention/$round"
[[ $(cat "$candidate/eval/status.txt") == PASS ]] || exit 2
fast_build=${H3_AUTOTUNE_ATTENTION_BUILD:-"$candidate/install/attention"}
model=${H3_AUTOTUNE_MODEL:-/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3}
destination="$framework/validation/$round"
mkdir -p -- "$framework/validation"
mkdir -- "$destination"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
code="$project/submission-xsh"
[[ -f "$code/optimization/adaln/launch.py" ]] || code="$project"
source "$code/optimization/load-candidate-config.sh" \
  "${H3_AUTOTUNE_BASE_CONFIG:-$project/optimization/selected-best.env}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
overlay=$(cat "$H3_ADALN_BUILD/native-overlay.txt")
export PYTHONPATH="$overlay:$PYTHONPATH"
command=(python -u "$IFAB_AFS_HOME/scripts/launch_h3.py"
  "$code/optimization/adaln/launch.py" --replay-run
  --model "$model"
  --captures "${H3_AUTOTUNE_CAPTURES:-$project/optimization/kernel-baseline/denoise-captures}"
  --out "$destination/replay" --steps 1,25,49 --iterations 3 --wire-ablation
  --fast-attention-build "$fast_build")
hostname > "$destination/command.log"
date -Iseconds >> "$destination/command.log"
printf '%q ' "${command[@]}" >> "$destination/command.log"
printf '\n' >> "$destination/command.log"
status=0
"${command[@]}" > "$destination/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$destination/exit-code"
exit "$status"
