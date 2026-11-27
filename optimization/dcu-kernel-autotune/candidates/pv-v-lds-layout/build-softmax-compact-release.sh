#!/usr/bin/env bash
# 从已冻结的 P 行布局 + V 预排布发布目录重建精简 softmax 候选。
set -eo pipefail
if [[ $# != 2 ]]; then
    printf '用法：build-softmax-compact-release.sh PREPACKED_RELEASE NEW_OUTPUT_DIR\n' >&2
    exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=$(cd -- "$scripts/../../../.." && pwd)
reference=$(realpath -- "$1")
out=$(realpath -m -- "$2")
test -s "$reference/candidate.env"
export H3_AUTOTUNE_CODE_ROOT="$project" IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
previous=$(python -c '
import json,sys
from pathlib import Path
d=json.load(open(sys.argv[1])); assert d["kind"] == "p-row-prepacked-v-v1"
print(Path(d["reference_release"]) / "attention")
' "$reference/release.json")
python "$scripts/summarize-public-a.py" check "$previous" "$reference/attention"
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/build-exit-code"' EXIT
python "$scripts/build-p-nan-payload.py" "$reference/attention" "$out/full-kv"
python "$scripts/build-softmax-ror4.py" "$out/full-kv" "$out/attention"
python "$scripts/summarize-public-a.py" check "$reference/attention" "$out/attention"
cp -- "$reference/candidate.env" "$out/candidate.env"
printf '\nexport H3_ATTENTION_AOT_BUILD=%q\n' "$out/attention" >> "$out/candidate.env"
python - "$reference" "$out" "$project" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
reference, out, project = map(Path, sys.argv[1:])
scripts = project / 'optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout'
sources = [scripts / name for name in (
    'build-softmax-compact-release.sh', 'build-p-nan-payload.py', 'build-softmax-ror4.py',
    'build-qk-online-order.py', 'summarize-public-a.py', 'summarize-softmax-dpp-steady.py')]
sources += [project / 'optimization/dcu-kernel-autotune/adapters/compare-h3-device.py']
files = ('dense-attention-packed.hsaco', 'dense-attention-aot.hsaco',
         'dense-attention-fallback.hsaco', 'dense-attention-packed-reference.hsaco',
         'prepack-value.hsaco', 'dense-attention-aot.so')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
record = dict(kind='p-row-prepacked-v-softmax-v1', reference_release=str(reference),
              reference_manifest_sha256=digest(reference / 'release.json'),
              candidate_config=str(out / 'candidate.env'),
              source_commit_base=subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip(),
              source_files={str(path.relative_to(project)): digest(path) for path in sources},
              artifacts={name: digest(out / 'attention' / name) for name in files},
              qualified=False,
              qualification_note='重建与冻结指令检查不代替完整模型质量及稳态性能验收')
(out / 'release.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
PY
printf '%s\n' "$out/candidate.env"
