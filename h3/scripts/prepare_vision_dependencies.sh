#!/usr/bin/env bash
# Install CPU image preprocessing support while retaining the vendor compute stack.
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/activate.sh"
export SONGSHAN_ROOT
set -u
case "${1:-}" in
    ""|--install-only) ;;
    *) echo 'Usage: bash scripts/prepare_vision_dependencies.sh [--install-only]' >&2; exit 2 ;;
esac
mkdir -p "$project_root/logs" "$project_root/artifacts"
exec > >(tee -a "$project_root/logs/prepare_vision_dependencies.log") 2>&1
python - <<'PY'
from importlib.metadata import version
assert version('torch') == '2.9.0+das.opt1.dtk2604'
assert version('triton') == '3.5.1+das.opt1.dtk2604.torch290'
PY
install_args=(--index-url https://download.pytorch.org/whl/cpu --only-binary=:all:
    -c "$project_root/env/inference-constraints.txt" 'torchvision==0.24.0+cpu')
python -m pip install --dry-run --report "$project_root/artifacts/vision-dependency-plan.json" "${install_args[@]}"
python - "$project_root/artifacts/vision-dependency-plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for entry in plan['install']:
    if entry['metadata']['name'].lower() != 'torchvision':
        raise SystemExit('Unexpected dependency change; stopped before install: ' + entry['metadata']['name'])
PY
python -m pip install --report "$project_root/artifacts/vision-dependency-install.json" "${install_args[@]}"
python - <<'PY'
import sys
from importlib.metadata import version
# The shell script can be invoked from any directory.
import os
sys.path.insert(0, os.environ['SONGSHAN_ROOT'])
from songshan_h3.preprocessing import check_vision_api
assert version('torch') == '2.9.0+das.opt1.dtk2604'
assert version('triton') == '3.5.1+das.opt1.dtk2604.torch290'
print(check_vision_api())
PY
if [[ "${1:-}" != --install-only && -f "$project_root/artifacts/model-download.json" &&
      -f "$project_root/samples/minimax_h3_public/case.json" ]]; then
    python "$project_root/scripts/check_model_components.py"
fi
echo 'Vision dependency checks passed; vendor Torch and Triton retained.'
