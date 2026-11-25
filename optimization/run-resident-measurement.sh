#!/usr/bin/env bash
# 常驻实例测速；输出目录由 Python 入口新建，显存采样覆盖完整进程。
set -eo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:?infinilm or indexed}
out=${2:?new absolute output directory}
shift 2
test ! -e "$out"
test ! -e "$out.exit-code"
test ! -e "$out.vram.jsonl"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
{
    hostname
    date -Is
    printf 'Slurm job %s\n' "${SLURM_JOB_ID:-direct}"
    env | sort | rg '^(H3_|NCCL_|RCCL_|TRITON_CACHE_DIR=|NINETOOTHED_CACHE_DIR=)' || true
    printf '%q ' bash "$project/optimization/run-sequence.sh" "$mode" --out "$out" "$@"
    printf '\n'
} > "$out.command.txt"
python "$project/optimization/sample-vram.py" \
    --output "$out.vram.jsonl" --until "$out.exit-code" > "$out.vram.log" 2>&1 &
sampler=$!
trap 'test -f "$out.exit-code" || printf "130\n" > "$out.exit-code"' EXIT
status=0
/usr/bin/time -o "$out.process-time.txt" -f 'process.wall %e s\nprocess.peak_rss %M KiB' \
    bash "$project/optimization/run-sequence.sh" "$mode" --out "$out" "$@" > "$out.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out.exit-code"
wait "$sampler" || printf 'VRAM sampler failed: %s\n' "$out.vram.log" >&2
exit "$status"
