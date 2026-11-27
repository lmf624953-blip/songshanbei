#!/usr/bin/env bash
# 当前已验收 137.344 秒版本的构建、常驻配对及普通入口复测。
set -eo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
scripts="$project/optimization/dcu-kernel-autotune/candidates/pv-v-lds-layout"
export H3_AUTOTUNE_CODE_ROOT="$project"

usage() {
    cat <<'EOF'
用法：
  bash optimization/reproduce-public-a.sh build NEW_WORK_DIR
  bash optimization/reproduce-public-a.sh resident RELEASE_DIR MODEL_DIR NEW_OUTPUT_DIR
  bash optimization/reproduce-public-a.sh single RELEASE_DIR MODEL_DIR NEW_OUTPUT_DIR

build 从源码生成 NEW_WORK_DIR/release，不运行模型。
resident 运行八次完整 A，排除双方各一次预热，汇总三组配对。
single 用新进程运行一次官方 A，另记进程总墙钟及原入口计时。
目录均使用绝对路径；NEW_WORK_DIR 和 NEW_OUTPUT_DIR 必须尚不存在。
先准备仓库根目录 runtime，并在独占八卡 BW1000 的干净 Bash 中运行。
EOF
}

if [[ $# == 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
mode=$1
shift
case "$mode" in
    build)
        [[ $# == 1 ]] || { usage >&2; exit 2; }
        work=$(realpath -m -- "$1")
        test -x "$project/runtime/.venv/bin/python"
        test -r "$project/runtime/artifacts/attention-bridge-build.json"
        mkdir -- "$work"
        trap 'printf "%s\n" "$?" > "$work/build-exit-code"' EXIT
        git -C "$project" rev-parse HEAD > "$work/source-commit.txt"
        git -C "$project" status --short > "$work/source-status.txt"
        printf '基础构建日志：%s\n' "$work/build.log"
        H3_BUILD_JOBS=${H3_BUILD_JOBS:-8} H3_AOT_SAVE_IR=1 \
            bash "$project/optimization/rebuild-v2.sh" "$work/build" \
            --short-attention --deterministic-keyframe --long-attention-q128-packed \
            --fused-wide-norm-indexed-original-serial-compact --fused-swiglu-row \
            > "$work/build.log" 2>&1
        bash "$scripts/build-row-sum-release.sh" \
            "$work/build/attention" "$work/build/candidate.env" "$work/row-sum" \
            > "$work/row-sum.log" 2>&1
        bash "$scripts/build-p-row-prepacked-release.sh" \
            "$work/row-sum" "$work/release" > "$work/release.log" 2>&1
        printf '已构建当前验收版：%s\n' "$work/release"
        ;;
    resident)
        [[ $# == 3 ]] || { usage >&2; exit 2; }
        test -s "$1/release.json" || { printf '需要当前组合版 release.json，请先执行 build。\n' >&2; exit 2; }
        exec bash "$project/optimization/benchmark-public-a.sh" "$@"
        ;;
    single)
        [[ $# == 3 ]] || { usage >&2; exit 2; }
        release=$(realpath -- "$1")
        model=$(realpath -- "$2")
        out=$(realpath -m -- "$3")
        test -r "$model/model_index.json"
        # 子 shell 的环境检查不计入普通入口进程墙钟。
        (
            export IFAB_AFS_HOME="$project/runtime"
            source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
            source "$project/optimization/load-candidate-config.sh" "$release/candidate.env"
            export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
            reference=$(python -c 'import json,sys; from pathlib import Path
d=json.load(open(sys.argv[1])); assert d["kind"] in ("p-row-prepacked-v-v1", "p-row-prepacked-v-softmax-v1")
print(Path(d["reference_release"]) / "attention")' "$release/release.json")
            python "$scripts/summarize-public-a.py" check "$reference" "$release/attention" \
                --first-frame "$IFAB_AFS_HOME/unified-dog-input/first_frame.png"
        )
        mkdir -- "$out"
        cp -- "$release/release.json" "$out/release.json"
        cp -- "$release/candidate.env" "$out/candidate.env"
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4
        command=(bash "$project/optimization/run-optimized.sh"
            --config "$release/candidate.env" fl2va --model-path "$model"
            --first-frame "$project/runtime/unified-dog-input/first_frame.png"
            --text "A dog falls asleep" --gpus 0,1,2,3,4,5,6,7
            --steps 50 --num-frames 124 --seed 42 --output "$out/video.mp4")
        {
            hostname
            date -Is
            git -C "$project" rev-parse HEAD
            printf '%q ' "${command[@]}"
            printf '\n'
        } > "$out/command.log"
        if command -v hy-smi >/dev/null 2>&1; then
            hy-smi > "$out/devices.log" 2>&1 || true
        fi
        printf '普通入口日志：%s\n' "$out/run.log"
        status=0
        /usr/bin/time -o "$out/process-time.txt" -f 'process.wall %e s' \
            "${command[@]}" > "$out/run.log" 2>&1 || status=$?
        printf '%s\n' "$status" > "$out/process-exit-code"
        cat "$out/process-time.txt"
        exit "$status"
        ;;
    *) usage >&2; exit 2 ;;
esac
