#!/usr/bin/env bash
# Compile the downloaded native framework sources for this host's Hygon stack.
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/activate.sh"
export SONGSHAN_ROOT
set -u

mode=build
case "${1:-}" in
    --check) mode=check ;;
    --configure-only) mode=configure ;;
    "") ;;
    *) echo "Usage: bash scripts/build_frameworks.sh [--check|--configure-only]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Unexpected extra arguments" >&2
    exit 2
fi
mkdir -p "$project_root/logs" "$project_root/artifacts"
exec > >(tee -a "$project_root/logs/build_frameworks.log") 2>&1
trap 'echo "Framework build failed at line $LINENO; see logs/build_frameworks.log" >&2' ERR

build_jobs="${SONGSHAN_BUILD_JOBS:-4}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "SONGSHAN_BUILD_JOBS must be a positive integer" >&2
    exit 2
fi

python "$project_root/scripts/check_inference_dependencies.py"
xmake --version
"$CUDA_HOME/bin/nvcc" --version
INFINILM_CXX11_ABI="$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
export INFINILM_CXX11_ABI
SONGSHAN_PYBIND11_INCLUDE="$(python -c 'import pybind11; print(pybind11.get_include())')"
export SONGSHAN_PYBIND11_INCLUDE
SONGSHAN_PYTHON_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
export SONGSHAN_PYTHON_INCLUDE
if [[ ! -f "$SONGSHAN_PYTHON_INCLUDE/Python.h" ]]; then
    echo "Missing Python development header: $SONGSHAN_PYTHON_INCLUDE/Python.h" >&2
    exit 1
fi

for spec in InfiniCore:f32e2fbb2a9b2ca912b5711f4f3836c403196a74 InfiniLM:65deb91f840ebaef752d028fad2f3217ce0950f4; do
    name="${spec%%:*}"
    expected="${spec#*:}"
    src="$SONGSHAN_STORAGE/sources/$name"
    if [[ "$(git -C "$src" rev-parse HEAD)" != "$expected" ]]; then
        echo "Unexpected $name revision; existing sources retained: $src" >&2
        exit 1
    fi
done

core_flags=(
    --hygon-dcu=y --hygon-arch="$HYGON_ARCH" --nv-gpu=n
    --ccl=y --aten=y --cudnn=n --ninetoothed=n --omp=n
    --cuda="$CUDA_HOME" --cc="$CC" --cxx="$CXX"
    --cxflags="-D_GLIBCXX_USE_CXX11_ABI=$INFINILM_CXX11_ABI"
    -m release
)

# Optional existing upstream integration. Do not point this at an unrelated
# NVIDIA wheel or pretend that building the base libraries registers H3 MHA.
if [[ -n "${SONGSHAN_FLASH_ATTN_ROOT:-}" ]]; then
    if [[ ! -f "$SONGSHAN_FLASH_ATTN_ROOT/flash_attn/lib/libflash_attention.so" ||
          ! -f "${FLASH_ATTN_2_CUDA_SO:-}" ]]; then
        echo "FlashAttention requires both the Hygon libflash_attention.so and FLASH_ATTN_2_CUDA_SO" >&2
        exit 1
    fi
    core_flags+=(--flash-attn="$SONGSHAN_FLASH_ATTN_ROOT")
fi

printf 'Build mode: %s; Hygon architecture: %s; C++ ABI: %s; jobs: %s\n' \
    "$mode" "$HYGON_ARCH" "$INFINILM_CXX11_ABI" "$build_jobs"
printf 'Core configuration: '
printf '%q ' "${core_flags[@]}"
printf '\n'
echo "This builds native libraries. H3 attention integration and competition operator audit remain separate checks."
if [[ "$mode" == check ]]; then
    echo "Build prerequisites passed. No framework source has been compiled."
    exit 0
fi

# Keep the original downloads intact and put all build/install side effects in
# a writable project-local copy. The .venv editable installs use these copies.
stage_source() {
    local src="$SONGSHAN_STORAGE/sources/$1"
    local dest="$project_root/sources/$1-build"
    local expected
    expected="$(git -C "$src" rev-parse HEAD)"
    if [[ ! -e "$dest" ]]; then
        local staging="$dest.staging.$$"
        mkdir -p "$project_root/sources"
        cp -a "$src" "$staging" || return 1
        mv "$staging" "$dest" || return 1
    fi
    if [[ "$(git -C "$dest" rev-parse HEAD)" != "$expected" ]]; then
        echo "Build copy has a different revision; retained for inspection: $dest" >&2
        return 1
    fi
    printf '%s\n' "$dest"
}

SONGSHAN_CORE_BUILD_DIR="$(stage_source InfiniCore)"
SONGSHAN_LM_BUILD_DIR="$(stage_source InfiniLM)"
export SONGSHAN_CORE_BUILD_DIR SONGSHAN_LM_BUILD_DIR
apply_source_patches() {
    local repo="$1" directory="$2"
    for patch_file in "$project_root"/patches/"$repo"/*.patch; do
        [[ -f "$patch_file" ]] || continue
        if git -C "$directory" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
            echo "$repo patch already applied: $(basename "$patch_file")"
        else
            git -C "$directory" apply --check "$patch_file"
            git -C "$directory" apply "$patch_file"
            echo "Applied $repo patch: $(basename "$patch_file")"
        fi
    done
}
apply_source_patches InfiniCore "$SONGSHAN_CORE_BUILD_DIR"
apply_source_patches InfiniLM "$SONGSHAN_LM_BUILD_DIR"
(
    cd "$SONGSHAN_CORE_BUILD_DIR"
    xmake f -y "${core_flags[@]}"
    if [[ "$mode" != configure ]]; then
        xmake build -y -j "$build_jobs"
        xmake install -y
        xmake build -y -j "$build_jobs" _infinicore
        xmake install -y _infinicore
        python -m pip install -c "$project_root/env/inference-constraints.txt" \
            --no-build-isolation -e .
    fi
)
if [[ "$mode" == configure ]]; then
    echo "InfiniCore configuration passed. InfiniLM configuration requires installed Core libraries."
    exit 0
fi
(
    cd "$SONGSHAN_LM_BUILD_DIR"
    xmake f -y -m release --cxx11-abi="$INFINILM_CXX11_ABI" --cc="$CC" --cxx="$CXX"
    xmake build -y -j "$build_jobs" _infinilm
    xmake install -y _infinilm
    python -m pip install -c "$project_root/env/inference-constraints.txt" \
        --no-build-isolation -e .
)
python "$project_root/scripts/check_frameworks.py"
python "$project_root/scripts/build_attention_bridge.py"
echo "Native framework build and shared-library checks passed. Device imports may remain pending; see artifacts/framework-build-current.json."
