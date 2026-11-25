#!/usr/bin/env bash
# 用独立普通 fl2va 入口完成公开 A，保持官方工作量并调用原始比较函数。
set -eo pipefail
adapter=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
framework=$(cd -- "$adapter/.." && pwd)
project=$(cd -- "$framework/../.." && pwd)
config=${1:?需要候选配置路径}
out=${2:?需要尚不存在的输出目录}
mkdir -- "$out"
config=$(realpath -- "$config")
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
code="$project/submission-xsh"
[[ -f "$code/optimization/run-optimized.sh" ]] || code="$project"
source "$code/optimization/load-candidate-config.sh" "$config"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
overlay=$(cat "$H3_ADALN_BUILD/native-overlay.txt")
export PYTHONPATH="$overlay:$code/optimization:$PYTHONPATH"
python - "$out" <<'PY'
from pathlib import Path
import sys
from source_snapshot import snapshot
snapshot(Path(sys.argv[1]))
PY
first="$IFAB_AFS_HOME/unified-dog-input/first_frame.png"
model=${H3_AUTOTUNE_MODEL:-/work1/share/sugonhpcapp01/zhanghengyu/songshanbei/models/MiniMax-H3}
command=(bash "$code/optimization/run-optimized.sh" --config "$config" fl2va
  --model-path "$model" --first-frame "$first" --text "A dog falls asleep"
  --gpus 0,1,2,3,4,5,6,7 --steps 50 --num-frames 124 --seed 42 --output "$out/video.mp4")
hostname > "$out/command.log"
date -Iseconds >> "$out/command.log"
printf '%q ' "${command[@]}" >> "$out/command.log"
printf '\n' >> "$out/command.log"
trap 'test -f "$out/exit-code" || printf "130\n" > "$out/exit-code"' EXIT
status=0
/usr/bin/time -o "$out/process-time.txt" -f 'process.wall %e s\nprocess.peak_rss %M KiB' \
  "${command[@]}" > "$out/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/exit-code"
[[ "$status" == 0 ]] || exit "$status"
status=0
python "$code/optimization/validate-official-output.py" "$out" \
  "${H3_AUTOTUNE_REFERENCE:-$IFAB_AFS_HOME/outputs/unified_fl2va/20261123_114003/reference.mp4}" "$first" \
  > "$out/validation.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/validation-exit-code"
exit "$status"
