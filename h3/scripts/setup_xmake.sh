#!/usr/bin/env bash
# Build against the host's glibc; prebuilt Linux bundles need newer glibc.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env/storage.sh"
mkdir -p "$project_root/artifacts" "$project_root/logs" "$project_root/.local/bin"

# Check that an existing executable runs, not merely that it exists in PATH.
if command -v xmake >/dev/null 2>&1 && \
        xmake --version > "$project_root/logs/xmake-version.log" 2>&1; then
    xmake_version="$(cat "$project_root/logs/xmake-version.log")"
    if [[ "$xmake_version" =~ v2\.9\.8([^0-9]|$) ]]; then
        cat "$project_root/logs/xmake-version.log"
        exit 0
    fi
    echo 'Existing xmake differs from v2.9.8; building the project-local pinned version.'
fi
echo "Preparing project-local xmake v2.9.8 against $(getconf GNU_LIBC_VERSION)."
if [[ -f "$project_root/logs/xmake-version.log" ]]; then
    cat "$project_root/logs/xmake-version.log"
fi

src="$SONGSHAN_STORAGE/sources/xmake"
if [[ ! -e "$src" ]]; then
    git clone --recursive --depth 1 --branch v2.9.8 \
        https://github.com/xmake-io/xmake.git "$src"
fi
expected="13fc392386960040c3da1ddcbd104423b02fbe1b"
actual="$(git -C "$src" rev-parse HEAD)"
if [[ "$actual" != "$expected" ]]; then
    echo "Existing xmake checkout is not v2.9.8; retained without changes: $src" >&2
    exit 1
fi
# A complete existing checkout can be built offline and may be read-only.
if [[ -n "$(git -C "$src" submodule status --recursive | sed -n '/^-/p')" ]]; then
    git -C "$src" submodule update --init --recursive
fi
git -C "$src" rev-parse HEAD > "$project_root/artifacts/xmake.commit"
git -C "$src" submodule status --recursive > "$project_root/artifacts/xmake.submodules"

# Configure/make write into their source tree. Keep the downloaded checkout
# intact and stage it into a project-local build tree when it is read-only.
if [[ ! -w "$src" ]]; then
    build_src="$project_root/sources/xmake-build"
    if [[ ! -e "$build_src" ]]; then
        mkdir -p "$project_root/sources"
        cp -a "$src" "$build_src"
    fi
    if [[ "$(git -C "$build_src" rev-parse HEAD)" != "$expected" ]]; then
        echo "Unexpected revision in build tree: $build_src" >&2
        exit 1
    fi
    src="$build_src"
fi

build_jobs="${SONGSHAN_BUILD_JOBS:-4}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "SONGSHAN_BUILD_JOBS must be a positive integer" >&2
    exit 1
fi
(
    cd "$src"
    ./configure
    make -j"$build_jobs"
    make install PREFIX="$project_root/.local"
)
hash -r
"$project_root/.local/bin/xmake" --version | tee "$project_root/logs/xmake-version.log"
echo "xmake built from source and installed in $project_root/.local."
