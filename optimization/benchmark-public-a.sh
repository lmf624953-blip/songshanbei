#!/usr/bin/env bash
# 在同一八卡进程中复测具有固定设备代码身份的长注意力版本。
set -eo pipefail
if [[ $# != 3 ]]; then
    printf '用法：bash optimization/benchmark-public-a.sh RELEASE_DIR MODEL_DIR NEW_OUTPUT_DIR\n' >&2
    exit 2
fi
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
release=$(realpath -- "$1")
model=$(realpath -- "$2")
out=$(realpath -m -- "$3")
summary="$project/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout/summarize-public-a.py"
test -r "$model/model_index.json"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$project/optimization/load-candidate-config.sh" "$release/candidate.env"
base_attention="$release/joint-kv"
if [[ -f "$release/release.json" ]]; then
    base_attention=$(python -c '
import json
from pathlib import Path
import sys
record = json.load(open(sys.argv[1]))
if record.get("kind") != "p-row-prepacked-v-v1":
    sys.exit("不支持的公开 A 配对发布类型")
print((Path(record["reference_release"]) / "attention").resolve(strict=True))
' "$release/release.json")
fi
export H3_ATTENTION_AOT_BUILD="$base_attention"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
python "$summary" check "$H3_ATTENTION_AOT_BUILD" "$release/attention" \
    --first-frame "$IFAB_AFS_HOME/unified-dog-input/first_frame.png"
mkdir -- "$out"

command=(bash "$project/optimization/run-sequence.sh" indexed
    --model "$model" --out "$out/sequence"
    --sequence A,A,A,A,A,A,A,A --fast-attention-build "$release/attention"
    --attention-variants base,fast,base,fast,fast,base,base,fast)
{
    hostname
    date -Is
    git -C "$project" rev-parse HEAD 2>/dev/null || true
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$out/command.log"
if command -v hy-smi >/dev/null 2>&1; then
    hy-smi > "$out/devices.log" 2>&1 || true
fi
printf '原始日志 %s\n' "$out/run.log"
status=0
/usr/bin/time -o "$out/process-time.txt" -f 'process.wall %e s' \
    "${command[@]}" > "$out/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/process-exit-code"
if [[ "$status" != 0 ]]; then
    exit "$status"
fi
python "$summary" report "$out/sequence" "$out/benchmark.json"
