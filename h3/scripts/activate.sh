#!/usr/bin/env bash
# Source this file from any working directory.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '请在当前终端执行：source %q\n' "${BASH_SOURCE[0]}" >&2
    echo '使用 bash 执行只会激活子进程，无法改变当前终端的环境。' >&2
    exit 2
fi
SONGSHAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$SONGSHAN_ROOT/env/machine.local.sh" ]]; then
    source "$SONGSHAN_ROOT/env/machine.local.sh" || return 1
fi
export DTK_ROOT="${DTK_ROOT:-/public/software/compiler/dtk-26.04}"
if [[ ! -f "$DTK_ROOT/env.sh" ]]; then
    echo "DTK env.sh missing: $DTK_ROOT" >&2
    return 1
fi
source "$DTK_ROOT/env.sh"
source "$SONGSHAN_ROOT/.venv/bin/activate"
source "$SONGSHAN_ROOT/env/storage.sh"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export HYGON_CUDA_HOME="$DTK_ROOT/cuda/cuda"
export CUDA_HOME="$HYGON_CUDA_HOME"
export PYTHON="$SONGSHAN_ROOT/.venv/bin/python"
export SONGSHAN_GCC_ROOT="${SONGSHAN_GCC_ROOT:-/public/software/compiler/gcc-11.4.0}"
export PATH="$SONGSHAN_GCC_ROOT/bin:$PATH"
# cc1/cc1plus also depend on the libraries shipped beside this GCC. Merely
# adding lib64 lets `gcc --version` work but fails as soon as compilation starts.
for SONGSHAN_GCC_LIB in lib64 isl/lib gmp/lib mpfr/lib mpc/lib; do
    if [[ -d "$SONGSHAN_GCC_ROOT/$SONGSHAN_GCC_LIB" ]]; then
        export LD_LIBRARY_PATH="$SONGSHAN_GCC_ROOT/$SONGSHAN_GCC_LIB:${LD_LIBRARY_PATH:-}"
    fi
done
unset SONGSHAN_GCC_LIB
export CC="$SONGSHAN_GCC_ROOT/bin/gcc"
export CXX="$SONGSHAN_GCC_ROOT/bin/g++"
export PYTHONNOUSERSITE=1
export INFINI_ROOT="$SONGSHAN_ROOT/.infini"
export PATH="$SONGSHAN_ROOT/.local/bin:$INFINI_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
# Torch bundles renamed libnl libraries; RCCL/verbs also request their system
# SONAMEs. Keep both requests on the same physical libraries to avoid duplicate
# registration lists and a crash in rtnl_tc_unregister at normal process exit.
# This creates only project-local aliases, without importing Torch or GPUs.
if ! "$PYTHON" "$SONGSHAN_ROOT/scripts/prepare_netlink_runtime.py" >/dev/null; then
    echo 'Unable to prepare the project libnl runtime aliases' >&2
    return 1
fi
export LD_LIBRARY_PATH="$SONGSHAN_ROOT/.local/lib/torch-netlink:$LD_LIBRARY_PATH"
export XMAKE_GLOBALDIR="$SONGSHAN_ROOT/.xmake"
export PIP_CACHE_DIR="$SONGSHAN_ROOT/.cache/pip"
export NINETOOTHED_CACHE_DIR="$SONGSHAN_ROOT/.cache/ninetoothed"
export TRITON_CACHE_DIR="$SONGSHAN_ROOT/.cache/triton"
# This vendor backend already provides an opt-in LLVM 17 compatibility pass.
# The Python attention entry also selects it, including in existing terminals.
export DPA4_TRITON_LLVM17_COMPAT=1
export DPA4_TRITON_IR_LOG="$SONGSHAN_ROOT/artifacts/triton-llvm17-ir"
