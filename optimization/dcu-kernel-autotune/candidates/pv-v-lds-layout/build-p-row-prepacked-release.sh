#!/usr/bin/env bash
# 从已核验的求和交错发布目录构建 P 行布局和 V 预排布组合。
set -eo pipefail
if [[ $# != 2 ]]; then
    printf '用法：build-p-row-prepacked-release.sh ROW_SUM_RELEASE NEW_OUTPUT_DIR\n' >&2
    exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=$(cd -- "$scripts/../../../.." && pwd)
reference=$(realpath -- "$1")
out=$(realpath -m -- "$2")
test -s "$reference/candidate.env"
export H3_AUTOTUNE_CODE_ROOT="$project"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
python "$scripts/summarize-public-a.py" check "$reference/joint-kv" "$reference/attention"
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/build-exit-code"' EXIT
python "$scripts/build-v-prepack-aot.py" "$out/prepack" --tile 2048 --warps 8
python "$scripts/build-prepacked-v.py" "$reference/attention" "$out/combined" \
    --position after-k-write --p-row-layout
python "$scripts/build-prepacked-dispatch.py" "$reference/attention" "$out/combined" \
    "$out/prepack" "$out/attention"
cp -- "$reference/candidate.env" "$out/candidate.env"
printf '\nexport H3_ATTENTION_AOT_BUILD=%q\n' "$out/attention" >> "$out/candidate.env"
python - "$reference" "$out" "$project" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
reference, out, project = map(Path, sys.argv[1:])
scripts = project / "optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout"
sources = [scripts / name for name in (
    "build-p-row-prepacked-release.sh", "summarize-public-a.py", "summarize-softmax-dpp-steady.py",
    "build-v-prepack-aot.py", "v_prepack_kernel.py", "build-prepacked-v.py", "build-p-row-lds.py",
    "build-disjoint-k-lds.py", "prepacked_attention_host.cc", "build-prepacked-dispatch.py",
    "prepacked_dispatch_extension.cc")]
sources += [project / "optimization/adaln" / name for name in
            ("aot_codegen.py", "elf_validation.py", "aot_runtime.py", "attention_buffer_host.cc")]
sources += [project / "optimization/dcu-kernel-autotune/adapters/compare-h3-device.py"]
files = ("dense-attention-packed.hsaco", "dense-attention-aot.hsaco",
         "dense-attention-fallback.hsaco", "dense-attention-packed-reference.hsaco",
         "prepack-value.hsaco", "dense-attention-aot.so")
record = dict(kind="p-row-prepacked-v-v1", reference_release=str(reference),
              candidate_config=str(out / "candidate.env"),
              source_commit_base=subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
              source_files={str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sources},
              artifacts={name: hashlib.sha256((out / "attention" / name).read_bytes()).hexdigest()
                         for name in files},
              qualified=False,
              qualification_note="重建入口，未在新环境自动执行模型验收；性能与质量以该环境的实际运行记录为据")
(out / "release.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
PY
printf '%s\n' "$out/candidate.env"
