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
            # 两侧都核对指针和布局。候选的只读位视图在预热时准备，
            # 继续引用同一存储，输入内容变化会立即反映到kernel。
            key = tuple((name, value.data_ptr(), tuple(value.shape), tuple(value.stride()))
                        if isinstance(value, torch.Tensor) else (name, value)
                        for name, value in values.items())
            if getattr(self, "_last_key", None) != key:
                prepared = dict(values)
                if self.meta.get("packed_bf16"):
                    assert values["offset"] % 2 == 0
                    for name in [*(f"p{i}" for i in range(8)), "residual", "gate", "output"]:
                        value = values[name]
                        assert value.dtype == torch.bfloat16 and value.stride(-1) == 1
                        assert value.data_ptr() % 4 == 0 and value.shape[-1] % 2 == 0
                        prepared[name] = value.view(torch.int32)
                    prepared["offset"] //= 2
                self._last_key, self._last_values = key, prepared
            return super().launch(stream, self._last_values)

    main_function.__globals__["Image"] = PackedImage
    main_function()


if __name__ == "__main__":
    main()
