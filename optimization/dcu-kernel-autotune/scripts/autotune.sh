#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_NAME="${PROFILE:-meam}"
PROFILE_FILE="${ROOT}/profiles/${PROFILE_NAME}.conf"
[[ -f "${PROFILE_FILE}" ]] || { echo "profile not found: ${PROFILE_FILE}" >&2; exit 1; }
source "${PROFILE_FILE}"
if [[ ${KOKKOS_ARCH_ARGS+x} != x ]]; then KOKKOS_ARCH_ARGS=(); fi
: "${RUN_SCRIPT:?profile must define RUN_SCRIPT}"
: "${CORRECT_SCRIPT:?profile must define CORRECT_SCRIPT}"
: "${METRIC_PARSER:?profile must define METRIC_PARSER}"
: "${BUILD_SCRIPT:?profile must define BUILD_SCRIPT}"

SOURCE_SNAPSHOTS_DIR="${ROOT}/source_snapshots/${ALGORITHM}"
BASELINE_SNAPSHOT="${SOURCE_SNAPSHOTS_DIR}/baseline"
ROUNDS_DIR="${ROOT}/rounds/${ALGORITHM}"
REPORT_DIR="${ROOT}/reports/${ALGORITHM}"
BASE_INSTALL="${ROUNDS_DIR}/baseline/install"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/${ALGORITHM}}"
ARTIFACT_TYPE="${ARTIFACT_TYPE:-executable}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-lmp}"
if [[ ${ARTIFACTS+x} != x ]]; then ARTIFACTS=("bin/${EXECUTABLE_NAME}"); fi
RUN_PREFIX="${RUN_PREFIX-mpirun -np 1}"
HIPPROF="${HIPPROF:-hipprof}"
E2E_METRIC="${E2E_METRIC:-ns/day}"
EFFECTIVE_THRESHOLD="${EFFECTIVE_THRESHOLD:-2.0}"
CORRECTNESS_RTOL="${CORRECTNESS_RTOL:-1e-4}"
CORRECTNESS_ATOL="${CORRECTNESS_ATOL:-1e-8}"
BUILD_JOBS="${BUILD_JOBS:-96}"
FOCUSED_PMC="${FOCUSED_PMC:-1}"
if [[ ${TRACE_ARGS+x} != x ]]; then TRACE_ARGS=(); fi
SUMMARY_METRIC_REGEX="${SUMMARY_METRIC_REGEX:-[0-9.]+ ns/day}"
SUMMARY_METRIC_SED="${SUMMARY_METRIC_SED:-s/ .*//}"
PMC_SAMPLE_REGEX="${PMC_SAMPLE_REGEX:-kernel time}"

log() { echo "[$1] ${*:2}"; }
die() { log ERROR "$*" >&2; exit 1; }
round_dir() { echo "${ROUNDS_DIR}/$1"; }
round_status() { cat "$(round_dir "$1")/eval/status.txt" 2>/dev/null || echo unknown; }
next_round() {
  local first_index="${ROUND_FIRST_INDEX:-1}"
  [[ "$first_index" =~ ^[0-9]+$ ]] || die "ROUND_FIRST_INDEX must be a non-negative integer"
  local max=$((first_index - 1)) d n
  for d in "${ROUNDS_DIR}"/round_*/; do
    [[ -d "$d" && "$(basename "$d")" =~ ^round_([0-9]+)$ ]] || continue
    n=$((10#${BASH_REMATCH[1]})); (( n > max )) && max=$n
  done
  printf 'round_%03d' $((max + 1))
}
ensure() { mkdir -p "$1"; }
install_dir() { echo "$(round_dir "$1")/install"; }
artifact_path() { echo "$(install_dir "$1")/$2"; }
assert_artifacts() {
  local round="$1" rel path
  [[ ${#ARTIFACTS[@]} -gt 0 ]] || die "ARTIFACTS is empty"
  for rel in "${ARTIFACTS[@]}"; do
    path="$(artifact_path "$round" "$rel")"
    [[ -f "$path" ]] || { log ERROR "build artifact missing: $path"; return 1; }
    [[ "$ARTIFACT_TYPE" != executable || -x "$path" ]] || { log ERROR "build artifact is not executable: $path"; return 1; }
  done
}
round_executable() {
  local round="$1"
  [[ "$ARTIFACT_TYPE" == executable ]] || return 0
  echo "$(artifact_path "$round" "bin/${EXECUTABLE_NAME}")"
}
load_env() {
  if [[ -n "${ENV_SCRIPT:-}" && -f "${ENV_SCRIPT}" ]]; then
    local nounset_was_set=0
    [[ $- == *u* ]] && nounset_was_set=1
    set +u
    source "${ENV_SCRIPT}"
    (( nounset_was_set )) && set -u
  fi
}

snapshot_init() {
  ensure "${BASELINE_SNAPSHOT}"
  local rel
  for rel in "${SNAPSHOT_FILES[@]}"; do
    [[ -f "${SOURCE_ROOT}/${rel}" ]] || die "source file not found: ${SOURCE_ROOT}/${rel}; update ${PROFILE_FILE}"
    ensure "${BASELINE_SNAPSHOT}/$(dirname "$rel")"
    cp -p "${SOURCE_ROOT}/${rel}" "${BASELINE_SNAPSHOT}/${rel}"
  done
  (
    cd "${BASELINE_SNAPSHOT}"
    find . -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum
  ) > "${BASELINE_SNAPSHOT}/manifest.sha256"
}
restore_source() {
  local rel
  # 注意：这里必须 return 而非 die。die 用 exit 终止整个 shell（不可被外层
  # `restore_source ... || snapshot_init` 捕获），导致新 profile 首次跑 baseline
  # 时 snapshot_init 永远不执行、静默失败。return 1 让裸调用仍由 set -e 中止，
  # 而 baseline() 的 `|| snapshot_init` 可以正常恢复。
  [[ -d "${BASELINE_SNAPSHOT}" ]] || { log WARN "no baseline snapshot; run init first"; return 1; }
  for rel in "${SNAPSHOT_FILES[@]}"; do
    [[ -f "${BASELINE_SNAPSHOT}/${rel}" ]] || die "snapshot missing: ${rel}"
    ensure "${SOURCE_ROOT}/$(dirname "$rel")"
    cp -p "${BASELINE_SNAPSHOT}/${rel}" "${SOURCE_ROOT}/${rel}"
  done
}
apply_delta() {
  local round="$1" f rel
  [[ -d "$(round_dir "$round")/source_delta" ]] || die "round not found: $round"
  while IFS= read -r -d '' f; do
    rel="${f#$(round_dir "$round")/source_delta/}"
    [[ "$rel" != /* && "$rel" != *../* ]] || die "invalid source_delta path: $rel"
    ensure "${SOURCE_ROOT}/$(dirname "$rel")"
    cp -p "$f" "${SOURCE_ROOT}/${rel}"
  done < <(find "$(round_dir "$round")/source_delta" -type f -print0)
}
build() {
  local round="$1"
  local install="$(round_dir "$round")/install"
  ensure "$install"
  rm -rf "${install:?}"/*
  load_env
  # 增量构建：保留 BUILD_DIR（不再清空），cmake --build 只重编译本轮变更文件。
  # restore_source/apply_delta 用 cp -p 保留源文件时间戳——若不 touch，源文件
  # mtime 可能旧于构建树里的 .o，make 会认为"未变更"而跳过重编译，导致新代码
  # 没被编进去（陈旧二进制）。故 build 前强制 touch 本轮修改过的文件。
  local rel f
  # touch 全部 SNAPSHOT_FILES：restore_source 每次都会把这些文件还原成 baseline，
  # cp -p 保留旧 mtime，若某轮 source_delta 只是子集，未 touch 的文件会被 make
  # 误判"未变更"而跳过重编译，留下上一轮的陈旧 .o。故无条件 touch 全部。
  for rel in "${SNAPSHOT_FILES[@]}"; do
    [[ -f "${SOURCE_ROOT}/${rel}" ]] && touch "${SOURCE_ROOT}/${rel}"
  done
  # 再 touch source_delta 里的文件，覆盖 delta 中可能超出 SNAPSHOT_FILES 的文件。
  if [[ -d "$(round_dir "$round")/source_delta" ]]; then
    while IFS= read -r -d '' f; do
      rel="${f#$(round_dir "$round")/source_delta/}"
      [[ -f "${SOURCE_ROOT}/${rel}" ]] && touch "${SOURCE_ROOT}/${rel}"
    done < <(find "$(round_dir "$round")/source_delta" -type f -print0)
  fi
  # BUILD_CLEAN=1 时强制全量重建（清空构建树）
  if [[ "${BUILD_CLEAN:-0}" == "1" ]]; then
    rm -rf "${BUILD_DIR:?}"
  fi
  ensure "${BUILD_DIR}"
  SOURCE_ROOT="${SOURCE_ROOT}" BUILD_DIR="${BUILD_DIR}" BUILD_JOBS="${BUILD_JOBS}" \
    ENV_SCRIPT="${ENV_SCRIPT:-}" PACKAGE_ARGS="${PACKAGE_ARGS[*]}" KOKKOS_ARCH_ARGS="${KOKKOS_ARCH_ARGS[*]}" \
    "${BUILD_SCRIPT}" "${install}" \
    > "$(round_dir "$round")/build/build.log" 2>&1 || return 1
  assert_artifacts "$round"
}
runtime_ld() {
  local install="$1"
  local paths=("${install}/lib64" "${install}/lib")
  [[ -n "${OPAL_PREFIX:-}" ]] && paths+=("${OPAL_PREFIX}/lib")
  if [[ -n "${DTKROOT:-}" ]]; then
    paths+=("${DTKROOT}/hip/lib" "${DTKROOT}/llvm/lib" "${DTKROOT}/lib" "${DTKROOT}/lib64")
  fi
  paths+=("/opt/hyhal/lib64" "/opt/hyhal/lib")
  [[ -n "${LD_LIBRARY_PATH:-}" ]] && paths+=("${LD_LIBRARY_PATH}")
  local IFS=:
  echo "${paths[*]}"
}
run_adapter() {
  local round="$1" log_file="$2" profiler="$3" stdout_file="$4" exe install
  install="$(install_dir "$round")"
  exe="$(round_executable "$round")"
  ensure "$(dirname "$log_file")"
  # A rerun must not inherit performance markers or error text from a prior
  # attempt at the same round/profiling stage.
  : > "$log_file"
  load_env
  ( cd "${CASE_DIR}" && \
    LD_LIBRARY_PATH="$(runtime_ld "$install")" \
    EXE="$exe" INSTALL_DIR="$install" ROUND="$round" ROUND_DIR="$(round_dir "$round")" \
    LOG_FILE="$log_file" CASE_DIR="${CASE_DIR}" RUN_PREFIX="${RUN_PREFIX-}" \
    BENCHMARK_STEPS="${BENCHMARK_STEPS-}" BENCHMARK_EPOCHS="${BENCHMARK_EPOCHS-}" \
    BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE-}" BENCHMARK_SEED="${BENCHMARK_SEED-}" \
    DEVICE_ID="${DEVICE_ID-}" \
    INPUT_FILE="${INPUT_FILE-}" RUN_FLAGS="${RUN_FLAGS-}" \
    PROFILER="$profiler" "${RUN_SCRIPT}" > "$stdout_file" 2>&1 )
}
run_case() {
  local round="$1" log_file="$2"
  run_adapter "$round" "$log_file" "" "${log_file}.stdout"
  [[ -s "$log_file" ]] || log WARN "run log is empty: $log_file"
}
e2e_round() {
  local round="$1" log_stem="${2:-run}"
  local d="$(round_dir "$round")"
  [[ -d "$d" ]] || die "round not found: $round"
  [[ "$log_stem" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid E2E log stem: $log_stem"
  restore_source
  [[ "$round" == baseline ]] || apply_delta "$round"
  assert_artifacts "$round"
  if ! run_case "$round" "$d/eval/${log_stem}.log"; then
    restore_source
    return 1
  fi
  restore_source
}
# Correctness is delegated to the profile's CORRECT_SCRIPT, which owns the
# software-specific output comparison and writes "verdict: PASS/FAIL" to $out.
correct() {
  local base="$1" cand="$2" out="$3"
  RTOL="${CORRECTNESS_RTOL}" ATOL="${CORRECTNESS_ATOL}" \
    "${CORRECT_SCRIPT}" "$base" "$cand" "$out"
}
# Return success only when the candidate completed its E2E run and its wall
# time is lower than the baseline by the profile-selected minimum percentage.
# This is intentionally evaluated before any candidate trace or PMC run: the
# profiler collections are diagnostic evidence for an E2E winner, not a
# substitute for end-to-end performance.
e2e_improved() {
  local base_log="$1" cand_log="$2" base_seconds cand_seconds improvement
  base_seconds="$(grep -E 'AUTOTUNE_E2E_WALL_SECONDS=[0-9.]+' "$base_log" 2>/dev/null | tail -1 | sed -E 's/.*=//')"
  cand_seconds="$(grep -E 'AUTOTUNE_E2E_WALL_SECONDS=[0-9.]+' "$cand_log" 2>/dev/null | tail -1 | sed -E 's/.*=//')"
  [[ "$base_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || { log WARN "E2E gate: baseline wall time missing: $base_log"; return 1; }
  [[ "$cand_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || { log WARN "E2E gate: candidate wall time missing: $cand_log"; return 1; }
  improvement="$(awk -v base="$base_seconds" -v cand="$cand_seconds" 'BEGIN { printf "%.6f", 100.0 * (base - cand) / base }')"
  if awk -v improvement="$improvement" -v threshold="${E2E_IMPROVEMENT_MIN_PCT:-0}" 'BEGIN { exit !(improvement > threshold) }'; then
    log INFO "E2E gate PASS: baseline=${base_seconds}s candidate=${cand_seconds}s improvement=${improvement}%"
    return 0
  fi
  log INFO "E2E gate FAIL: baseline=${base_seconds}s candidate=${cand_seconds}s improvement=${improvement}% (requires >${E2E_IMPROVEMENT_MIN_PCT:-0}%)"
  return 1
}
profile_round() {
  local round="$1" kernel out
  [[ "$FOCUSED_PMC" == "1" ]] || return 0
  for kernel in "${KERNELS[@]}"; do
    out="$(round_dir "$round")/pmc/${kernel}"
    ensure "$(round_dir "$round")/pmc"
    run_adapter "$round" "$(round_dir "$round")/eval/${kernel}.log" \
      "${HIPPROF} --pmc --kernel-name ${kernel} -o ${out}" \
      "$(round_dir "$round")/pmc/${kernel}.stdout" || log WARN "profile failed: $kernel"
  done
}
# Step 2 of the round protocol: hipprof trace -> hotspot kernel names + % of GPU time.
trace_round() {
  local round="$1" d="$(round_dir "$1")" out trace_csv
  ensure "$d/trace"
  out="$d/trace/hotspot"
  rm -f "${out}.db" "${out}.db-journal" "${out}.hipkernel.csv" \
    "${out}.hiptrace.csv" "${out}.kernel.csv" "${out}.csv" "${out}.txt"
  run_adapter "$round" "$d/trace/hotspot.log" "${HIPPROF} ${TRACE_ARGS[*]} -o ${out}" "$d/trace/hotspot.stdout" || { log WARN "trace failed: $round"; return 0; }
  for trace_csv in "$out.kernel.csv" "$out.hipkernel.csv" "$out.csv"; do
    [[ -f "$trace_csv" ]] || continue
    python3 "${ROOT}/scripts/analyze_pmc.py" trace "$trace_csv" 12 > "$d/trace/hotspot.txt"
    return 0
  done
  log WARN "trace kernel csv not found for prefix: $out"
}
# Step 3 of the round protocol: full PMC (all kernels) + resource stats for target kernels.
pmc_full_round() {
  local round="$1" d="$(round_dir "$1")" out pmc_steps saved_steps
  ensure "$d/pmc"
  out="$d/pmc/full"
  rm -f "${out}.db" "${out}.db-journal" "${out}.txt" "${out}.csv" \
    "${out}.hipkernel.csv" "${out}.hiptrace.csv"
  pmc_steps="${PMC_STEPS:-${BENCHMARK_STEPS}}"
  saved_steps="${BENCHMARK_STEPS}"
  BENCHMARK_STEPS="${pmc_steps}"
  if ! run_adapter "$round" "$d/eval/pmc_full.log" "${HIPPROF} --pmc -o ${out}" "$d/pmc/full.stdout"; then
    BENCHMARK_STEPS="${saved_steps}"
    log WARN "pmc full failed: $round"
    return 0
  fi
  BENCHMARK_STEPS="${saved_steps}"
  local pmc_file="$out.txt"
  [[ -f "$pmc_file" ]] || pmc_file="$out.csv"
  if [[ -f "$pmc_file" ]]; then
    python3 "${ROOT}/scripts/analyze_pmc.py" pmc "$pmc_file" "${KERNELS[@]}" > "$d/pmc/full_summary.txt"
  else
    log WARN "pmc output not found: $out"
  fi
}
# Step 4 of the round protocol: E2E + per-kernel PMC comparison vs baseline.
metrics_round() {
  local round="$1" d="$(round_dir "$1")" base="$(round_dir baseline)"
  [[ -d "$d" ]] || die "round not found: $round"
  [[ -f "$d/eval/run.log" ]] || { log WARN "metrics: no run.log for $round"; return 0; }
  [[ -f "$d/pmc/full.txt" || -f "$d/pmc/full.csv" ]] || log WARN "metrics: no PMC output for $round"
  METRIC_PARSER="${METRIC_PARSER}" E2E_METRIC="${E2E_METRIC}" \
    python3 "${ROOT}/scripts/metrics.py" "$d" "$base" "${KERNELS[@]}"
}
new_round() {
  local desc="" kernel="${KERNELS[0]}"; while (($#)); do case "$1" in --name) desc="$2"; shift 2;; --kernel) kernel="$2"; shift 2;; *) die "unknown option: $1";; esac; done
  local round="$(next_round)"; local d="$(round_dir "$round")"; ensure "$d/source_delta"; ensure "$d/build"; ensure "$d/eval"; ensure "$d/pmc"
  local rn="${round#round_}"; rn=$((10#${rn}))
  cat > "$d/report.md" <<EOF
description: $desc
kernel: $kernel
algorithm: $ALGORITHM

## 🚀 Round $rn: [优化点名称]

### 1. 修改说明与代码变更
- **优化点**：
- **核心修改**：（关键代码 Diff / 函数块）

### 2. 正确性校验
- **测试结果**：[PASS / FAIL]
- **数值误差**：（与 Baseline 的能量/力误差对比）

### 3. 性能分析与硬件指标（PMC）
- **耗时 (ms)**：上一最优 ____ ms → 本轮 ____ ms（提升/下降 __%）
- **VGPR 占用**：____ 寄存器/Workitem
- **Register Spilling (Stack/Local Mem)**：____ 字节
- **Occupancy / Wavefront 并发度**：____%

### 4. 决策与后续规划
- **结论**：[保留为当前最优 (Best) / 性能无提升，放弃回滚]
- **当前历史最优版本**：Round ____
EOF
  log INFO "created $round; put edits in $d/source_delta"
}
baseline() {
  # baseline 必须自足：不要求先跑 init。这里确保 baseline round 的
  # build/eval/pmc 目录都存在（init 命令也会创建同样的目录）。
  restore_source 2>/dev/null || snapshot_init
  ensure "${BASE_INSTALL}"
  ensure "$(round_dir baseline)/build"
  ensure "$(round_dir baseline)/eval"
  ensure "$(round_dir baseline)/pmc"
  build baseline
  run_case baseline "$(round_dir baseline)/eval/run.log"
  if [[ "${BASELINE_SELF_CHECK:-0}" == "1" ]]; then
    correct "$(round_dir baseline)/eval/run.log" "$(round_dir baseline)/eval/run.log" "$(round_dir baseline)/eval/correctness.txt"
  fi
  profile_round baseline
  trace_round baseline
  pmc_full_round baseline
  echo baseline > "$(round_dir baseline)/eval/status.txt"
  report
}
run_round() {
  # 异常退出也恢复隔离源码，避免编译或剖析失败遗留候选改动。
  trap restore_source EXIT
  local round="$1"; local d="$(round_dir "$round")"; [[ -d "$d" ]] || die "round not found: $round"; restore_source; apply_delta "$round"
  if ! build "$round"; then echo build_failed > "$d/eval/status.txt"; restore_source; return 1; fi
  if [[ "${VALIDATION_ORDER:-e2e_correct_kernel}" == "correct_e2e_kernel" ]]; then
    # Correctness and E2E are separate launches. A numerical failure never
    # reaches a performance claim; kernel/PMC collection follows E2E only.
    if ! run_case "$round" "$d/eval/correctness_run.log"; then echo run_failed > "$d/eval/status.txt"; restore_source; return 1; fi
    if ! correct "$(round_dir baseline)/eval/run.log" "$d/eval/correctness_run.log" "$d/eval/correctness.txt"; then
      echo correctness_failed > "$d/eval/status.txt"
      log ERROR "correctness FAIL for $round; skipping e2e/trace/pmc/metrics"
      restore_source
      return 1
    fi
    if ! run_case "$round" "$d/eval/run.log"; then echo run_failed > "$d/eval/status.txt"; restore_source; return 1; fi
    if ! e2e_improved "$(round_dir baseline)/eval/run.log" "$d/eval/run.log"; then
      echo e2e_not_improved > "$d/eval/status.txt"
      log INFO "E2E did not improve for $round; skipping trace/PMC/metrics"
      restore_source
      report
      return 0
    fi
    trace_round "$round" || true
    pmc_full_round "$round" || true
    echo PASS > "$d/eval/status.txt"
    metrics_round "$round" || true
    restore_source
    report
    return 0
  fi
  if [[ "${VALIDATION_ORDER:-e2e_correct_kernel}" == "correct_kernel_e2e" ]]; then
    # First inspect the candidate's kernel shape and PMC evidence, then make
    # the independent E2E measurement. This keeps the hardware diagnosis
    # available even when E2E does not improve.
    if ! run_case "$round" "$d/eval/correctness_run.log"; then echo run_failed > "$d/eval/status.txt"; restore_source; return 1; fi
    if ! correct "$(round_dir baseline)/eval/run.log" "$d/eval/correctness_run.log" "$d/eval/correctness.txt"; then
      echo correctness_failed > "$d/eval/status.txt"
      log ERROR "correctness FAIL for $round; skipping trace/PMC/E2E"
      restore_source
      return 1
    fi
    trace_round "$round" || true
    pmc_full_round "$round" || true
    if ! run_case "$round" "$d/eval/run.log"; then echo run_failed > "$d/eval/status.txt"; restore_source; return 1; fi
    if e2e_improved "$(round_dir baseline)/eval/run.log" "$d/eval/run.log"; then
      echo PASS > "$d/eval/status.txt"
    else
      echo e2e_not_improved > "$d/eval/status.txt"
    fi
    metrics_round "$round" || true
    restore_source
    report
    return 0
  fi
  if ! run_case "$round" "$d/eval/run.log"; then echo run_failed > "$d/eval/status.txt"; restore_source; return 1; fi
  if ! correct "$(round_dir baseline)/eval/run.log" "$d/eval/run.log" "$d/eval/correctness.txt"; then
    echo correctness_failed > "$d/eval/status.txt"
    log ERROR "correctness FAIL for $round; skipping trace/pmc/metrics"
    restore_source
    return 1
  fi
  echo PASS > "$d/eval/status.txt"
  trace_round "$round" || true
  pmc_full_round "$round" || true
  metrics_round "$round" || true
  restore_source
  report
}
context() {
  local round="$1"; local d="$(round_dir "$round")"; [[ -d "$d" ]] || die "round not found: $round"
  { echo "# Agent Context: $ALGORITHM / $round"; echo; cat "$d/report.md"; echo; echo "## Source delta"; find "$d/source_delta" -type f -print | sort; echo; echo "## Current status"; round_status "$round"; echo; echo "## Baseline metrics"; grep -hE 'kernel time|vgpr count|sgpr count|scratch size' "$(round_dir baseline)/pmc"/* 2>/dev/null | head -40 || true; echo; echo "## Instructions"; echo "Make one small, measurable change for the target kernel. Preserve numerical correctness. Run: PROFILE=$PROFILE_NAME ./scripts/autotune.sh run $round"; } > "$d/agent_context.md"
  cat "$d/agent_context.md"
}
report() {
  ensure "$REPORT_DIR"
  local out="$REPORT_DIR/summary.md"
  {
    echo "# $ALGORITHM 优化汇总"
    echo
    if [[ -n "${CURRENT_BEST_ROUND:-}" ]]; then
      echo "当前最优版本：\`${CURRENT_BEST_ROUND}\`（\`$(round_status "$CURRENT_BEST_ROUND")\`）。${CURRENT_BEST_NOTE:-}详细交接见 \`handoff.md\`。"
      echo
    fi
    echo "| 轮次 | 状态 | $E2E_METRIC | kernel 时间样本数 |"
    echo '|---|---|---|---|'
    for d in "${ROUNDS_DIR}"/*/; do
      [[ -d "$d" ]] || continue
      local r="$(basename "$d")" status metric n
      [[ -f "$d/eval/status.txt" ]] || continue
      status="$(round_status "$r")"
      metric="$(grep -hE "$SUMMARY_METRIC_REGEX" "$d/eval/run.log" 2>/dev/null | tail -1 | sed -E "$SUMMARY_METRIC_SED" || true)"
      [[ -n "$metric" ]] || metric="-"
      # Pre-gate profiler artifacts are retained for diagnosis but are never
      # presented as evidence for a rejected candidate.
      if [[ "$status" == baseline || "$status" == PASS ]]; then
        n="$({ grep -hE "$PMC_SAMPLE_REGEX" "$d/pmc"/* 2>/dev/null || true; } | wc -l)"
      else
        n="-"
      fi
      echo "| $r | $status | $metric | $n |"
    done
  } > "$out"
  cp "$out" "$REPORT_DIR/summary.txt"
  log INFO "report: $out"
}
status() { echo "profile=$PROFILE_NAME algorithm=$ALGORITHM"; echo "source=$SOURCE_ROOT"; echo "case=$CASE_DIR"; echo "kernels=${KERNELS[*]}"; for d in "${ROUNDS_DIR}"/*/; do [[ -d "$d" ]] && echo "$(basename "$d"): $(round_status "$(basename "$d")")"; done; }
diff_source() { local rel; for rel in "${SNAPSHOT_FILES[@]}"; do diff -u "${BASELINE_SNAPSHOT}/${rel}" "${SOURCE_ROOT}/${rel}" || true; done; }
check() {
  local err=0
  [[ -n "${SOURCE_ROOT:-}" && -d "${SOURCE_ROOT}" ]] || { log ERROR "SOURCE_ROOT missing/not a dir: ${SOURCE_ROOT:-}"; err=1; }
  [[ -n "${CASE_DIR:-}" && -d "${CASE_DIR}" ]] || { log ERROR "CASE_DIR missing/not a dir: ${CASE_DIR:-}"; err=1; }
  [[ -x "${BUILD_SCRIPT}" ]] || { log ERROR "BUILD_SCRIPT not executable: ${BUILD_SCRIPT}"; err=1; }
  [[ -x "${RUN_SCRIPT}" ]] || { log ERROR "RUN_SCRIPT not executable: ${RUN_SCRIPT}"; err=1; }
  [[ -x "${CORRECT_SCRIPT}" ]] || { log ERROR "CORRECT_SCRIPT not executable: ${CORRECT_SCRIPT}"; err=1; }
  [[ -f "${METRIC_PARSER}" ]] || { log ERROR "METRIC_PARSER not found: ${METRIC_PARSER}"; err=1; }
  if [[ -n "${ENV_SCRIPT:-}" ]]; then [[ -f "${ENV_SCRIPT}" ]] || { log ERROR "ENV_SCRIPT not found: ${ENV_SCRIPT}"; err=1; }; fi
  [[ ${#SNAPSHOT_FILES[@]} -gt 0 ]] || { log ERROR "SNAPSHOT_FILES is empty"; err=1; }
  [[ ${#KERNELS[@]} -gt 0 ]] || { log ERROR "KERNELS is empty"; err=1; }
  [[ ${#ARTIFACTS[@]} -gt 0 ]] || { log ERROR "ARTIFACTS is empty"; err=1; }
  (( err )) && return 1
  log INFO "profile OK: algorithm=$ALGORITHM source=$SOURCE_ROOT case=$CASE_DIR kernels=${#KERNELS[@]}"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  init) ensure "$ROUNDS_DIR/baseline/build"; ensure "$ROUNDS_DIR/baseline/eval"; ensure "$ROUNDS_DIR/baseline/pmc"; snapshot_init; log INFO "initialized $ALGORITHM";;
  baseline) baseline;; new|new-round) new_round "$@";; run) [[ $# -eq 1 ]] || die 'usage: run ROUND'; run_round "$1";;
  build)
    [[ $# -eq 1 ]] || die 'usage: build ROUND'
    trap restore_source EXIT
    restore_source
    [[ "$1" == baseline ]] || apply_delta "$1"
    build "$1"
    restore_source
    ;;
  e2e) [[ $# -ge 1 && $# -le 2 ]] || die 'usage: e2e ROUND [LOG_STEM]'; e2e_round "$@";;
  profile) [[ $# -eq 1 ]] || die 'usage: profile ROUND'; restore_source; [[ "$1" == baseline ]] || apply_delta "$1"; profile_round "$1"; restore_source;;
  trace) [[ $# -eq 1 ]] || die 'usage: trace ROUND'; restore_source; [[ "$1" == baseline ]] || apply_delta "$1"; trace_round "$1"; restore_source;;
  pmc) [[ $# -eq 1 ]] || die 'usage: pmc ROUND'; restore_source; [[ "$1" == baseline ]] || apply_delta "$1"; pmc_full_round "$1"; restore_source;;
  correct) [[ $# -eq 1 ]] || die 'usage: correct ROUND'; if correct "$(round_dir baseline)/eval/run.log" "$(round_dir "$1")/eval/run.log" "$(round_dir "$1")/eval/correctness.txt"; then echo PASS > "$(round_dir "$1")/eval/status.txt"; else echo correctness_failed > "$(round_dir "$1")/eval/status.txt"; fi;;
  context) [[ $# -eq 1 ]] || die 'usage: context ROUND'; context "$1";; metrics) [[ $# -eq 1 ]] || die 'usage: metrics ROUND'; metrics_round "$1";; report) report;; status) status;; restore) restore_source;; diff) diff_source;; check) check;;
  help|*) echo "Usage: PROFILE=meam $0 {init|baseline|new|context|run|build|e2e|profile|trace|pmc|correct|metrics|report|status|restore|diff|check}";;
esac
