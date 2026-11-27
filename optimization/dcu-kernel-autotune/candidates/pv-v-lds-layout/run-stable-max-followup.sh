#!/usr/bin/env bash
# 等本轮完整八卡验收实际退出，再在单卡串行验证后续候选。
set -eo pipefail
experiment=$(realpath -- "${1:?需要本轮实验目录}")
code=$(realpath -- "${2:?需要源码仓库}")
qualifier_pid=${3:?需要正在运行的完整验收PID}
out="$experiment/stable-max-verification"
scripts="$code/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout"
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/exit-code"' EXIT
printf '等待完整验收进程 %s 退出；此阶段不创建GPU上下文。\n' "$qualifier_pid"
while kill -0 "$qualifier_pid" 2>/dev/null; do
    sleep 10
done
# 父进程意外终止而模型子进程仍活跃时，也不能与其重叠测速。
python3 - "$experiment/qualification" <<'PYWAIT'
from pathlib import Path
import sys
import time
needle = sys.argv[1].encode()
while True:
    active = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / 'cmdline').read_bytes()
        except (OSError, PermissionError):
            continue
        if b'launch_h3.py' in command and needle in command:
            active.append(process.name)
    if not active:
        break
    time.sleep(10)
PYWAIT
python3 - "$experiment" "$code" <<'PYCHECK'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
qualification = root / 'qualification'
assert (qualification / 'qualification-exit-code').read_text().strip() == '0'
assert (qualification / 'qualified.json').is_file(), '完整数值与规格验收尚未成功'
assert len(json.loads((qualification / 'steady-paired/complete.json').read_text())['sequence']) == 11
source = json.loads((root / 'stable-max-builds/preparation.json').read_text())['source_sha256']
code = Path(sys.argv[2]) / 'optimization/dcu-kernel-autotune'
for name, expected in source.items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected, name
PYCHECK
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code" HIP_VISIBLE_DEVICES=0
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$experiment/reference.env"
python "$scripts/verify-p-nan-payload-edges.py" \
    "$H3_ATTENTION_AOT_BUILD" \
    "$experiment/builds" "$out/payload-edges.jsonl"
python "$scripts/verify-stable-max-values.py" "$experiment/rebuilt/combined" \
    "$experiment/stable-max-builds" "$out/value-edges.jsonl"
for mode in deferred guard-scale guarded; do
    python "$scripts/verify-p-nan-payload.py" "$experiment/rebuilt/combined" \
        "$experiment/stable-max-builds/$mode" "$out/$mode.jsonl"
done
printf '后续候选逐位校验及两个生产长度配对已完成；仍需按完整请求门槛评估。\n'
