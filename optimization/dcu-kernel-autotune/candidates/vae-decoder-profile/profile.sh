#!/usr/bin/env bash
# 在加载项目运行环境后执行，接受 hipprof 自动切分输出。
set -eo pipefail
if [[ $# -ne 2 ]]; then
    printf '用法: %s MODEL NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi
candidate=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=$(realpath -e "$1")
destination=$(realpath -m "$2")
mkdir -- "$destination"
trap 'printf "%s\n" "$?" > "$destination/exit-code"' EXIT
export HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4
unset H3_VAE_CACHE_REPLICAS
hipprof --hip-trace --group-stream --devices 0,1 --flush-interval 100 --buffer-size 10000 \
    --trace-off --output-type 0 -o "$destination/hipprof" \
    python -u "$candidate/capture.py" "$model" "$destination/capture.json" > "$destination/run.log" 2>&1
shopt -s nullglob
traces=("$destination"/hipprof_stream*.json)
(( ${#traces[@]} > 0 ))
for trace in "${traces[@]}"; do test -s "$trace"; done
test -s "$destination/capture.json"
