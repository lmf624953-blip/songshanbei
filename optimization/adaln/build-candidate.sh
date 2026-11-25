#!/usr/bin/env bash
# Build the experimental InfiniLM library against the pinned runtime's InfiniCore.
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=$(cd -- "$here/../.." && pwd)
dest=${1:?New build output directory required}
variant=${2:-adaln}
jobs=${H3_BUILD_JOBS:-32}
host_opt=${H3_NATIVE_CXX_OPTIMIZE:-default}
case "$host_opt" in
  default|O3) ;;
  *) printf 'Unsupported host optimization: %s\n' "$host_opt" >&2; exit 2;;
esac
case "$variant" in
  adaln|indexed|dense|precision|encoder-precision|wide-norm|encoder-prefill|short-attention) ;;
  *) printf 'Unknown variant: %s\n' "$variant" >&2; exit 2;;
esac
commit=$(git -C "$project" rev-parse --verify '45b2c3094d8d189e030f8984d7c07dead4db690a^{commit}')
mkdir -- "$dest"
dest=$(cd -- "$dest" && pwd)
scratch=$(mktemp -d /tmp/songshan-adaln-XXXXXXXX)
printf '%s\n' "$scratch" > "$dest/scratch-path"
printf '%s\n' "$commit" > "$dest/source-commit"
git -C "$project" archive "$commit" h3/frameworks/xmake h3/frameworks/InfiniLM |
  tar -xf - -C "$scratch"
source_tree="$scratch/h3/frameworks"
cd "$source_tree/InfiniLM"
git apply "$here/infinilm-adaln.patch"
if [[ "$variant" != adaln ]]; then
  git apply "$here/infinilm-indexed.patch"
fi
if [[ "$variant" != adaln && "$variant" != indexed ]]; then
  git apply "$here/infinilm-dense.patch"
fi
if [[ "$variant" == precision || "$variant" == encoder-precision || "$variant" == wide-norm || "$variant" == encoder-prefill ]]; then
  git apply "$here/infinilm-precision.patch"
fi
if [[ "$variant" == encoder-precision || "$variant" == wide-norm || "$variant" == encoder-prefill ]]; then
    git apply "$here/infinilm-encoder-precision.patch"
fi
if [[ "$variant" == wide-norm || "$variant" == encoder-prefill ]]; then
    git apply "$here/infinilm-wide-norm.patch"
fi
if [[ "$variant" == encoder-prefill ]]; then
    git apply "$here/infinilm-encoder-prefill.patch"
fi
if [[ "$variant" == short-attention ]]; then
    git apply "$here/infinilm-short-attention.patch"
fi
if [[ "$host_opt" == O3 ]]; then
  python3 - <<'PY'
from pathlib import Path
path = Path("xmake.lua")
text = path.read_text()
marker = 'target("_infinilm")'
if text.count(marker) != 1:
    raise RuntimeError("Unexpected InfiniLM xmake target")
path.write_text(text.replace(marker, 'add_cxxflags("-O3", {force = true})\n\n' + marker))
PY
fi
printf '%s\n' "$host_opt" > "$dest/host-optimization.txt"

module load compiler/dtk/26.04 compiler/gcc/11.4.0
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/activate.sh"
cd "$source_tree/xmake"
export XMAKE_PROGRAM_DIR="$PWD/xmake"
export XMAKE_GLOBALDIR="$scratch/xmake-global"
{
  bash configure
  make -j"$jobs"
  ./build/xmake --version
} > "$dest/xmake.log" 2>&1
xmake="$PWD/build/xmake"
export PATH="$(dirname "$xmake"):$PATH"
export SONGSHAN_PYBIND11_INCLUDE=$(python -c 'import pybind11; print(pybind11.get_include())')
export SONGSHAN_PYTHON_INCLUDE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
export INFINILM_CXX11_ABI=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')

cd "$source_tree/InfiniLM"
{
  "$xmake" f -y --ccache=n -m release --cxx11-abi="$INFINILM_CXX11_ABI" --cc="$CC" --cxx="$CXX"
  "$xmake" build -y -j"$jobs" _infinilm
  "$xmake" install -y _infinilm
} > "$dest/infini-build.log" 2>&1
python "$here/build-host.py" "$dest" > "$dest/host-build.log" 2>&1
python - "$project" "$dest" "$PWD" "$variant" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
import sysconfig
import tarfile

project, build, source = map(Path, sys.argv[1:4])
variant = sys.argv[4]
package = Path(sysconfig.get_path("purelib")) / "infinilm"
filename = "_infinilm" + sysconfig.get_config_var("EXT_SUFFIX")
overlay = build / "overlay"
shutil.copytree(package, overlay / "infinilm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", filename))
target = overlay / "infinilm/lib" / filename
shutil.copy2(source / "python/infinilm/lib" / filename, target)
files = {
    "base_native": package / "lib" / filename,
    "candidate_native": target,
    "host_shim": build / "adaln-host.so",
    "patch": project / "optimization/adaln/infinilm-adaln.patch",
    "kernel": project / "optimization/adaln/adaln_kernel.py",
    "xmake": source / "xmake.lua",
}
if variant != "adaln":
    files["indexed_patch"] = project / "optimization/adaln/infinilm-indexed.patch"
    files["indexed_kernel"] = project / "optimization/adaln/indexed_kernel.py"
if variant not in {"adaln", "indexed"}:
    files["dense_patch"] = project / "optimization/adaln/infinilm-dense.patch"
if variant in {"precision", "encoder-precision", "wide-norm", "encoder-prefill"}:
    files["precision_patch"] = project / "optimization/adaln/infinilm-precision.patch"
if variant in {"encoder-precision", "wide-norm", "encoder-prefill"}:
    files["encoder_precision_patch"] = project / "optimization/adaln/infinilm-encoder-precision.patch"
if variant in {"wide-norm", "encoder-prefill"}:
    files["wide_norm_patch"] = project / "optimization/adaln/infinilm-wide-norm.patch"
if variant == "encoder-prefill":
    files["encoder_prefill_patch"] = project / "optimization/adaln/infinilm-encoder-prefill.patch"
if variant == "short-attention":
    files["short_attention_patch"] = project / "optimization/adaln/infinilm-short-attention.patch"
record = {}
for name, path in files.items():
    with path.open("rb") as stream:
        record[name] = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
if record["base_native"]["sha256"] == record["candidate_native"]["sha256"]:
    raise RuntimeError("Candidate library matches the pinned baseline library")
(build / "candidate-build.json").write_text(json.dumps(record, indent=2) + "\n")
(build / "native-overlay.txt").write_text(str(overlay) + "\n")
with tarfile.open(build / "InfiniLM-applied-source.tar.gz", "w:gz") as archive:
    def keep_source(info):
        if (any(part in {"build", ".xmake", "__pycache__"} for part in Path(info.name).parts)
                or info.name.endswith((".so", ".pyc"))):
            return None
        return info
    archive.add(source, arcname="InfiniLM", filter=keep_source)
PY
