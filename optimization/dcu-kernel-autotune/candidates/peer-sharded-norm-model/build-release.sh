#!/usr/bin/env bash
# 从已验收peer参考构建完整模型候选，包含运行组件；不自动改默认。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code=$(cd -- "$here/../../../.." && pwd)
reference=$(realpath -- "${1:?需要已验收peer候选配置}")
work=$(realpath -m -- "${2:?需要尚不存在的工作目录}")
xmake=$(realpath -- "${3:?需要xmake二进制}")
test ! -e "$work"
test -x "$xmake"
export IFAB_AFS_HOME="$code/runtime" H3_AUTOTUNE_CODE_ROOT="$code"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$reference"
test -z "${H3_PEER_SHARDED_NORM_BUILD:-}"
test -n "${H3_PEER_GATE_BUILD:-}"
mkdir -- "$work"
trap 'printf "%s\n" "$?" > "$work/build-exit-code"' EXIT
python "$here/build-native.py" "$H3_ADALN_BUILD" "$work/native" --xmake "$xmake" --jobs "${H3_BUILD_JOBS:-8}"
python "$here/../peer-sharded-norm/build-row-gate.py" "$work/nt" --tile 1024 --warps 4
python "$here/../peer-sharded-norm/build.py" "$work/nt" "$work/prototype" --split-arrivals
python "$here/build-bridge.py" "$work/prototype" "$work/model"
export H3_ADALN_BUILD="$work/native" H3_PEER_SHARDED_NORM_BUILD="$work/model"
python "$here/package-runtime.py" "$work/release"
