#!/usr/bin/env python3
"""Compare NineToothed tiles and HIP compiler options on one Hygon device."""
import argparse
import ctypes
from functools import partial
import json
from pathlib import Path
from statistics import median

import ninetoothed
import torch

from songshan_h3.attention_kernel import definition, kernel
from songshan_h3.compiler_compat import configure_hygon_compiler


def arrangement(query, key, value, is_causal, scale, output, *, key_block, query_block=64):
    def query_or_output(tensor):
        arranged = tensor.tile((1, 1, query_block, 128)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1)
        )
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    query_arranged = query_or_output(query)

    def key_or_value(tensor):
        arranged = tensor.tile((1, 1, key_block, 128)).tile((1, 1, -1, -1)).expand(
            (-1, -1, query_arranged.shape[-2], -1)
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    return (query_arranged, key_or_value(key), key_or_value(value),
            is_causal, scale, query_or_output(output))


def timings(call, iterations):
    call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    measured = []
    for _ in range(iterations):
        start.record()
        call()
        end.record()
        end.synchronize()
        measured.append(start.elapsed_time(end))
    return measured


def compiler_variants():
    return (
        {"waves_per_eu": 2}, {"waves_per_eu": 3}, {"waves_per_eu": 4},
        {"matrix_instr_nonkdim": 16}, {"matrix_instr_nonkdim": 32}, {"kpack": 2},
        {"schedule_hint": "attention"},
        {"schedule_hint": "attention", "waves_per_eu": 2},
    )


def tune_compiler(handle, options):
    generated = handle._kernel
    run = generated.run
    observed = {}

    def configured(*args, **kwargs):
        compiled = run(*args, **kwargs, **options)
        if compiled is not None and not observed:
            observed.update((option, getattr(compiled.metadata, option)) for option in options)
        return compiled

    generated.run = configured
    return observed


def native_launcher(build):
    from adaln.aot_runtime import load_attention

    native = load_attention(build).songshan_bench_attention
    native.argtypes = ([ctypes.c_int] * 2 + [ctypes.c_float] + [ctypes.c_void_p] * 4
                       + [ctypes.c_int] * 4 + [ctypes.c_void_p])
    native.restype = ctypes.c_char_p

    def launch(q, k, v, scale, out):
        tensors = (q, k, v, out)
        stream = torch.cuda.current_stream(q.device)
        error = native(q.device.index, q.shape[2], scale, *(t.data_ptr() for t in tensors),
                       *(t.stride(2) for t in tensors), stream.cuda_stream)
        if error:
            raise RuntimeError(error.decode())

    return launch


def run(args):
    generator = torch.Generator(device="cuda").manual_seed(42)
    dtype = getattr(torch, args.dtype)

    def tensor(sequence, heads):
        return torch.randn((1, sequence, heads, args.head_dim), dtype=dtype,
                           generator=generator, device="cuda").transpose(1, 2)

    q, k, v = (tensor(args.q_len, args.q_heads), tensor(args.k_len, args.kv_heads),
               tensor(args.k_len, args.kv_heads))
    if args.fused_value_stride:
        fused = torch.empty((1, args.k_len, 3 * args.kv_heads, args.head_dim),
                            dtype=dtype, device="cuda")
        strided = fused[:, :, -args.kv_heads:].transpose(1, 2)
        strided.copy_(v)
        v = strided
    scale = args.head_dim ** -0.5
    reference, result = torch.empty_like(q), torch.empty_like(q)
    baseline = kernel()
    baseline_times = timings(lambda: baseline(q, k, v, args.causal, scale, reference), args.iterations)
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError("Pinned kernel returned non-finite outputs")
    records = [{"query_block": 64, "key_block": 64, "num_warps": 4, "pinned": True,
                "median_ms": median(baseline_times), "samples_ms": baseline_times}]
    if args.aot:
        launch = native_launcher(args.aot.resolve())
        samples = timings(lambda: launch(q, k, v, scale, result), args.iterations)
        error = (result.float() - reference.float()).abs()
        record = {"build": str(args.aot), "median_ms": median(samples), "samples_ms": samples,
                  "exact": bool(torch.equal(reference, result)), "max_abs_error": float(error.max()),
                  "mean_abs_error": float(error.mean()), "all_finite": bool(torch.isfinite(result).all())}
        records.append(record)
        print(record, flush=True)
        variants = ()
    elif args.compiler_sweep:
        variants = ((64, 64, 4, options) for options in compiler_variants())
    else:
        if args.extended_query_sweep:
            tiles = ((128, 64, 4), (256, 64, 4), (128, 64, 2), (64, 64, 1))
        else:
            tiles = ((32, 64, 4), (64, 64, 2), (128, 64, 8)) if args.query_sweep else (
                (64, 32, 4), (64, 128, 4), (64, 128, 8)
            )
        variants = ((*tile, {}) for tile in tiles)
    for index, (query_block, key_block, num_warps, options) in enumerate(variants):
        record = {"query_block": query_block, "key_block": key_block,
                  "num_warps": num_warps, "pinned": False, "compiler_options": options}
        try:
            _, application, params = definition()
            launch = ninetoothed.make(
                partial(arrangement, key_block=key_block, query_block=query_block), application, params,
                backend="triton", platform="hygon-bw1000", num_warps=num_warps,
                num_stages=1, max_num_configs=1,
                kernel_name=f"h3_attention_q{query_block}_k{key_block}_w{num_warps}_{index}",
            )
            observed = tune_compiler(launch, options) if options else {}
            times = timings(
                lambda: launch(q, k, v, args.causal, scale, result), args.iterations,
            )
            if observed != options:
                raise RuntimeError(f"Applied compiler options {observed} differ from {options}")
            difference = (result.float() - reference.float()).abs()
            record.update(
                samples_ms=times, median_ms=median(times),
                all_finite=bool(torch.isfinite(result).all()),
                exact=bool(torch.equal(result, reference)),
                max_abs_error=float(difference.max()), mean_abs_error=float(difference.mean()),
            )
        except Exception as error:
            record["error"] = repr(error)
        records.append(record)
        print(record, flush=True)
    after = timings(lambda: baseline(q, k, v, args.causal, scale, reference), args.iterations)
    records[0].update(median_after_ms=median(after), samples_after_ms=after)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", required=True, type=int)
    parser.add_argument("--k-len", required=True, type=int)
    parser.add_argument("--q-heads", required=True, type=int)
    parser.add_argument("--kv-heads", required=True, type=int)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--query-sweep", action="store_true")
    sweep.add_argument("--extended-query-sweep", action="store_true")
    sweep.add_argument("--compiler-sweep", action="store_true")
    sweep.add_argument("--aot", type=Path)
    parser.add_argument("--fused-value-stride", action="store_true")
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.q_len, args.k_len, args.q_heads, args.kv_heads, args.iterations) < 1:
        parser.error("Lengths, heads and iteration count must be positive")
    if args.q_heads % args.kv_heads or (args.causal and args.q_len > args.k_len):
        parser.error("Invalid grouped-head or causal shape")
    if args.aot and (args.causal or args.q_len != args.k_len or args.q_heads != 7 or args.kv_heads != 7
                     or args.head_dim != 128 or args.dtype != "bfloat16"):
        parser.error("AOT expects equal noncausal lengths, 7 heads and BF16 width 128")
    configure_hygon_compiler()
    records = run(args)
    with args.output.open("x") as stream:
        json.dump({"shape": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items() if key != "output"},
                   "configurations": records}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
