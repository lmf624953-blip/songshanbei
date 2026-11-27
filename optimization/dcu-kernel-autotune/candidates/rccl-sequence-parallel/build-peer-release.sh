#!/usr/bin/env bash
# 从冻结的前序配置构建九齿镜像、通信桥和独立native，暂不选为正式默认。
set -eo pipefail
if [[ $# != 3 ]]; then
    printf '用法：build-peer-release.sh REFERENCE_CONFIG NEW_OUTPUT XMAKE_BINARY\n' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code=$(cd -- "$here/../../../.." && pwd)
reference=$(realpath -- "$1")
out=$(realpath -m -- "$2")
xmake=$(realpath -- "$3")
test -x "$xmake"
test ! -e "$out"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$reference"
if [[ -n "${H3_PEER_GATE_BUILD:-}" ]]; then
    printf '前序配置已含peer回调，请传入第一次接入前冻结的reference.env\n' >&2
    exit 2
fi
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/build-exit-code"' EXIT
# 冻结已求值配置，避免候选再次source后来被切换的selected-best.env。
python - "$out/reference.env" <<'PYCONFIG'
import os,shlex,sys
from pathlib import Path
names=('H3_ADALN_AOT_BUILD','H3_ADALN_BUILD','H3_ATTENTION_AOT_BUILD','H3_ENCODER_INITIAL_CACHE',
       'H3_ENCODER_TP_COUNT','H3_INDEXED_BUILD','H3_PY_ATTENTION_FP32_SCALE','H3_QKNORM_BUILD',
       'H3_QKNORM_MROPE_BUILD','H3_SHORT_ATTENTION_AOT_BUILD','H3_SWIGLU_BUILD',
       'H3_VAE_DETERMINISTIC_ENCODE','H3_VAE_DEVICES','H3_WIDE_NORM_INDEXED_BUILD')
Path(sys.argv[1]).write_text('# 已求值的前序正式配置；不追踪未来selected-best变化。\n'+
    ''.join('export '+name+'='+shlex.quote(os.environ[name])+'\n' for name in names if name in os.environ))
PYCONFIG
python "$here/build-peer-nt.py" "$out/nt" --tile 1024 --warps 4
python "$here/build-peer-bridge.py" "$out/nt" "$out/peer"
python "$here/build-peer-native.py" "$H3_ADALN_BUILD" "$out/native" \
    --xmake "$xmake" --jobs "${H3_BUILD_JOBS:-8}"
{
    printf '# 九齿分片通信/门控候选；资格以独立完整请求记录为准。\n'
    printf 'source %q\n' "$out/reference.env"
    printf 'export H3_ADALN_BUILD=%q\n' "$out/native"
    printf 'export H3_PEER_GATE_BUILD=%q\n' "$out/peer"
    printf 'export H3_PEER_GATE_ENABLED=1\n'
} > "$out/candidate.env"
python - "$out" "$code" "$reference" <<'PYRELEASE'
import hashlib,json,subprocess,sys
from pathlib import Path
out,code,reference=map(Path,sys.argv[1:])
here=code/'optimization/dcu-kernel-autotune/candidates/rccl-sequence-parallel'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(kind='h3-peer-nt-indexed-gate-v1',qualified=False,
    qualification_note='构建完成不代替真实模型、完整官方入口及稳态质量性能验收。',
    reference_config=str(reference),reference_config_sha256=digest(reference),
    candidate_config=str(out/'candidate.env'),source_commit=subprocess.check_output(
        ['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
    source_files={str(p.relative_to(code)):digest(p) for p in here.iterdir()
                  if p.is_file() and p.suffix in {'.py','.cc','.hip','.patch','.json','.sh'}},
    peer_build=json.loads((out/'peer/peer-runtime-build.json').read_text()),
    native_build=json.loads((out/'native/candidate-build.json').read_text()))
(out/'release.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
PYRELEASE
printf '%s\n' "$out/candidate.env"
