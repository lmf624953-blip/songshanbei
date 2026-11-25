#!/usr/bin/env bash
# 从冻结的 v3 AOT 适配源码建立隔离实验树，不导入其他项目的历史轮次。
set -euo pipefail
adapter=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
framework=$(cd -- "$adapter/.." && pwd)
project=$(cd -- "$framework/../.." && pwd)
source_dir="$framework/work/source"
snapshot="$framework/source_snapshots/h3-long-attention/baseline"
if [[ -e "$source_dir" || -e "$snapshot" ]]; then
  printf '实验源码或快照已存在，请复用已有轮次；需要恢复时使用 autotune.sh restore。\n' >&2
  exit 2
fi
test -d "$project/runtime"
mkdir -p -- "$source_dir"
cp -a -- "$framework/templates/h3-source/." "$source_dir/"
ln -s -- "$project/runtime" "$source_dir/runtime"
PROFILE=h3-long-attention bash "$framework/scripts/autotune.sh" init
