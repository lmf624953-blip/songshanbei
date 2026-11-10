#!/usr/bin/env bash
# Reuse the exact environment and runner validated on the platform.
set -eo pipefail
cd "${IFAB_AFS_HOME:?Set IFAB_AFS_HOME to the prepared project root}"
source scripts/unified_dog_env.sh
python -u scripts/run_unified_dog.py
