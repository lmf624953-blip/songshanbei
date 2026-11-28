#!/usr/bin/env bash
# 单独采集完整投影和通信重叠投影。正式性能使用 bench.py 的无 profiler 配对。
set -eo pipefail
if [[ $# != 3 ]]; then
    printf '用法: %s PIPELINE_SO NORM_BUILD NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi
library=$(realpath -e "$1")
norm=$(realpath -e "$2")
destination=$(realpath -m "$3")
candidate=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -- "$destination"
trap 'printf "%s\n" "$?" > "$destination/exit-code"' EXIT
for mode in full split_overlap; do
    hipprof --pmc --pmc-off --devices 0,1,2,3,4,5,6,7 -o "$destination/$mode" \
        python -u "$candidate/bench.py" "$library" "$norm" "$destination/$mode.json" \
        --rows 27570 --m 2688 --modes full split_overlap --profile-mode "$mode" \
        > "$destination/$mode.log" 2>&1
    test -s "$destination/$mode.txt"
done
