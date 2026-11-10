#!/usr/bin/env bash
# Network access is required. Never reset or overwrite existing checkouts.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env/storage.sh"
mkdir -p "$SONGSHAN_STORAGE/sources" "$project_root/artifacts"
for spec in InfiniCore:minimax-h3 InfiniLM:minimax_h3 ninetoothed:master ntops:master; do
    repo="${spec%%:*}"
    branch="${spec#*:}"
    revision="$(python3 - "$project_root/env/source-pins.json" "$repo" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]]['commit'])
PY
)"
    dest="$SONGSHAN_STORAGE/sources/$repo"
    if [[ -e "$dest" ]]; then
        echo "Existing checkout retained: $dest; inspect it before updating." >&2
        actual="$(git -C "$dest" rev-parse HEAD)"
        if [[ "$actual" != "$revision" ]]; then
            echo "Expected $repo commit $revision, found $actual in $dest; no checkout changed." >&2
            exit 1
        fi
    else
        git clone --no-checkout --branch "$branch" "https://github.com/InfiniTensor/$repo.git" "$dest"
        git -C "$dest" checkout --detach "$revision"
        git -C "$dest" submodule update --init --recursive
    fi
    git -C "$dest" rev-parse HEAD > "$project_root/artifacts/$repo.commit"
    git -C "$dest" submodule status --recursive > "$project_root/artifacts/$repo.submodules"
done
