#!/usr/bin/env bash
# 在已适配的 BW1000 运行目录上重新构建 v2 的所有原生和九齿产物。
set -eo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$(realpath -m -- "${1:?Provide a new build directory}")
variant=dense
deterministic=0
fused_qknorm_mrope=0
fused_wide_norm_indexed=0
wide_norm_mode=naive
attention_options=()
attention_query_block=32
attention_warps=2
for option in "${@:2}"; do
  case "$option" in
    --short-attention) variant=short-attention ;;
    --deterministic-keyframe) deterministic=1 ;;
    --fused-qknorm-mrope) fused_qknorm_mrope=1 ;;
    --fused-wide-norm-indexed) fused_wide_norm_indexed=1; fused_qknorm_mrope=1; wide_norm_mode=naive ;;
    --fused-wide-norm-indexed-original-serial) fused_wide_norm_indexed=1; fused_qknorm_mrope=1; wide_norm_mode=original-serial ;;
    --long-attention-lds) attention_options=(--optimize-long-attention) ;;
    --long-attention-q64) attention_options=(--optimize-long-attention); attention_query_block=64; attention_warps=4 ;;
    --long-attention-q64-full-load) attention_options=(--optimize-long-attention --split-kv-tail --elide-full-kv-load-mask); attention_query_block=64; attention_warps=4 ;;
    --long-attention-q64-buffer) attention_options=(--optimize-long-attention --split-kv-tail --elide-full-kv-load-mask --guarded-buffer-ops); attention_query_block=64; attention_warps=4 ;;
    --long-attention-q64-vwide) attention_options=(--optimize-long-attention --split-kv-tail --elide-full-kv-load-mask --guarded-buffer-ops --coalesce-pv-value); attention_query_block=64; attention_warps=4 ;;
    --long-attention-q64-qdirect) attention_options=(--optimize-long-attention --split-kv-tail --elide-full-kv-load-mask --guarded-buffer-ops --coalesce-pv-value --direct-attention-query); attention_query_block=64; attention_warps=4 ;;
    --long-attention-q64-packed) attention_options=(--optimize-long-attention --split-kv-tail --elide-full-kv-load-mask --guarded-buffer-ops --coalesce-pv-value --direct-attention-query --packed-attention-strides); attention_query_block=64; attention_warps=4 ;;
    --long-attention-q128-packed) attention_options=(--optimize-long-attention --split-kv-tail --elide-full-kv-load-mask --guarded-buffer-ops --coalesce-pv-value --direct-attention-query --packed-attention-strides); attention_query_block=128; attention_warps=4 ;;
    --long-attention-q64-tail) attention_options=(--optimize-long-attention --split-kv-tail); attention_query_block=64; attention_warps=4 ;;
    *) printf 'Usage: rebuild-v2.sh BUILD_DIR [--short-attention] [--deterministic-keyframe] [--fused-qknorm-mrope] [--fused-wide-norm-indexed | --fused-wide-norm-indexed-original-serial] [--long-attention-lds | --long-attention-q64 | --long-attention-q64-tail | --long-attention-q64-full-load | --long-attention-q64-buffer | --long-attention-q64-vwide | --long-attention-q64-qdirect | --long-attention-q64-packed | --long-attention-q128-packed]\n' >&2; exit 2 ;;
  esac
done
if [[ "$fused_qknorm_mrope" == 1 ]]; then variant=qknorm-mrope; fi
if [[ "$fused_wide_norm_indexed" == 1 ]]; then variant=wide-norm-indexed; fi
mkdir -- "$out"
export IFAB_AFS_HOME="$project/runtime"
source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh"
source_dir="$project/optimization/adaln"
H3_NATIVE_CXX_OPTIMIZE=O3 bash "$source_dir/build-candidate.sh" "$out/infinilm" "$variant"
python "$source_dir/build-aot.py" "$out/adaln"
python "$source_dir/build-indexed.py" "$out/indexed" --tile 2048 --warps 4
python "$source_dir/build-qk-norm.py" "$out/qknorm" --cub-order --warps 2
if [[ "$fused_qknorm_mrope" == 1 ]]; then
  python "$source_dir/build-qk-norm-mrope.py" "$out/qknorm-mrope"
fi
if [[ "$fused_wide_norm_indexed" == 1 ]]; then
  python "$source_dir/build-wide-norm-indexed.py" "$out/wide-norm-indexed" --mode "$wide_norm_mode"
fi
python "$source_dir/build-attention-aot.py" "$out/attention" \
  --fp32-scale --aligned-strides --query-block "$attention_query_block" --key-block 32 \
  --warps "$attention_warps" --stages 1 --kpack 2 "${attention_options[@]}"
if [[ "$variant" == short-attention || "$variant" == qknorm-mrope || "$variant" == wide-norm-indexed ]]; then
  python "$source_dir/build-attention-aot.py" "$out/short-attention" --fp32-scale --aligned-strides
fi
{
  printf 'export H3_ADALN_BUILD=%q\n' "$out/infinilm"
  printf 'export H3_ADALN_AOT_BUILD=%q\n' "$out/adaln"
  printf 'export H3_INDEXED_BUILD=%q\n' "$out/indexed"
  printf 'export H3_QKNORM_BUILD=%q\n' "$out/qknorm"
  if [[ "$fused_qknorm_mrope" == 1 ]]; then
    printf 'export H3_QKNORM_MROPE_BUILD=%q\n' "$out/qknorm-mrope"
  fi
  if [[ "$fused_wide_norm_indexed" == 1 ]]; then
    printf 'export H3_WIDE_NORM_INDEXED_BUILD=%q\n' "$out/wide-norm-indexed"
  fi
  printf 'export H3_ATTENTION_AOT_BUILD=%q\n' "$out/attention"
  if [[ "$variant" == short-attention || "$variant" == qknorm-mrope || "$variant" == wide-norm-indexed ]]; then
    printf 'export H3_SHORT_ATTENTION_AOT_BUILD=%q\n' "$out/short-attention"
  fi
  printf 'export H3_VAE_DEVICES=0,1\n'
  printf 'export H3_PY_ATTENTION_FP32_SCALE=1\n'
  printf 'export H3_ENCODER_TP_COUNT=4\n'
  printf 'export H3_ENCODER_INITIAL_CACHE=512\n'
  if [[ "$deterministic" == 1 ]]; then
    printf 'export H3_VAE_DETERMINISTIC_ENCODE=1\n'
  fi
} > "$out/candidate.env"
cp -- "$out/infinilm/source-commit" "$out/source-commit"
date -Is > "$out/build.complete"
printf '%s\n' "$out/candidate.env"
