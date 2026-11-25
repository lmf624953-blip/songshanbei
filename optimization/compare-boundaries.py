#!/usr/bin/env python3
"""在 CPU 上比较诊断采集的首次去噪输入及输出，定位重复请求的差异。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from leaves(value[key], f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}/{index}")
    else:
        yield prefix, value


def digest(tensor):
    # Byte view handles BF16 without converting its values or invoking a GPU.
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left, right):
    left, right = dict(leaves(left)), dict(leaves(right))
    if left.keys() != right.keys():
        raise RuntimeError("Captured structures differ")
    result = {}
    for key, a in left.items():
        b = right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            record = {"shape": list(a.shape), "dtype": str(a.dtype),
                      "same_layout": a.shape == b.shape and a.dtype == b.dtype}
            if record["same_layout"]:
                record.update(left_sha256=digest(a), right_sha256=digest(b),
                              exact=torch.equal(a, b), differing_elements=int((a != b).sum()))
                if a.is_floating_point() and a.numel():
                    delta = a.double() - b.double()
                    record.update(all_finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                                  max_abs_error=float(delta.abs().max()),
                                  rmse=float(delta.square().mean().sqrt()))
            else:
                record.update(right_shape=list(b.shape), right_dtype=str(b.dtype), exact=False)
            result[key] = record
        else:
            # Captures are trusted local artifacts. Metadata should consist of scalar values.
            result[key] = {"left": a, "right": b, "exact": type(a) is type(b) and a == b}
    return {"all_exact": all(row["exact"] for row in result.values()), "leaves": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    left, right = (torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                   for path in (args.left, args.right))
    result = {"left": str(args.left.resolve()), "right": str(args.right.resolve()),
              "step": [left["step"], right["step"]],
              "inputs": compare(left["inputs"], right["inputs"]),
              "outputs": compare(left["outputs"], right["outputs"])}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: result[key]["all_exact"] for key in ("inputs", "outputs")}))


if __name__ == "__main__":
    main()
