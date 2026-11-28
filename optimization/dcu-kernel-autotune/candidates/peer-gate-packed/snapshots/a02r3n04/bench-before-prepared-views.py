#!/usr/bin/env python3
"""复用八卡分片gate检查，以只读32位视图传递成对BF16数据。"""

from pathlib import Path
import runpy

import torch


def main():
    previous = Path(__file__).resolve().parent.parent / "peer-gate-metadata/bench.py"
    main_function = runpy.run_path(str(previous))["main"]
    original = main_function.__globals__["Image"]

    class PackedImage(original):
        def launch(self, stream, values):
            if self.meta.get("packed_bf16"):
                assert values["offset"] % 2 == 0
                values = dict(values)
                for name in [*(f"p{i}" for i in range(8)), "residual", "gate", "output"]:
                    value = values[name]
                    assert value.dtype == torch.bfloat16 and value.stride(-1) == 1
                    assert value.data_ptr() % 4 == 0 and value.shape[-1] % 2 == 0
                    values[name] = value.view(torch.int32)
                values["offset"] //= 2
            return super().launch(stream, values)

    main_function.__globals__["Image"] = PackedImage
    main_function()


if __name__ == "__main__":
    main()
