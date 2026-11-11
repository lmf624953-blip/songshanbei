#!/usr/bin/env bash
set -euo pipefail
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${IFAB_AFS_HOME:?Set IFAB_AFS_HOME to your personal writable AFS directory}"
phase="${1:-all}"
args=()
if [[ "$phase" == prepare || "$phase" == all ]]; then args+=(--download-inputs); fi
exec python3.11 "$code_root/rebuild.py" \
  --destination "${H3_INSTALL_ROOT:-$IFAB_AFS_HOME/songshan_h3_runtime}" \
  --cache-root "${H3_INPUT_CACHE:-$IFAB_AFS_HOME/songshan_h3_inputs}" \
  --phase "$phase" "${args[@]}"
