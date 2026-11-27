#!/usr/bin/env bash
# 单实例8次完整公开A；汇总字段wall.candidate.mean与完整资格验收一致。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code=$(cd -- "$here/../../../.." && pwd)
release=$(realpath -- "${1:?需要发布目录}")
model=$(realpath -- "${2:?需要权重目录}")
out=$(realpath -m -- "${3:?需要新输出目录}")
test -r "$model/model_index.json"
test ! -e "$out"
export IFAB_AFS_HOME="$code/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source "$code/optimization/load-candidate-config.sh" "$release/candidate.env"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
python - "$here" "$release/release.json" <<'PYCHECK'
import json,os,sys
sys.path.insert(0,sys.argv[1])
from peer_identity import check
assert json.load(open(sys.argv[2]))['kind']=='h3-peer-nt-indexed-gate-v1'
check(os.environ['H3_PEER_GATE_BUILD'],os.environ['H3_ADALN_BUILD'])
PYCHECK
mkdir -- "$out"
trap 'printf "%s\n" "$?" > "$out/exit-code"' EXIT
cp -- "$release/release.json" "$out/release.json"
bash "$code/optimization/run-resident-measurement.sh" indexed "$out/sequence" \
    --model "$model" --sequence A,A,A,A,A,A,A,A \
    --peer-gate-variants off,on,off,on,on,off,off,on
python "$here/report-peer-benchmark.py" "$out/sequence" "$out/benchmark.json"
