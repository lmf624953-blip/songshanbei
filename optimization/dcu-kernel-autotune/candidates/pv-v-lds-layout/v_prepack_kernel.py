"""用九齿重排 V 的完整 32-token 块，保留 BF16 原始位模式。"""

from functools import cache

import ninetoothed as nt
import ninetoothed.language as ntl

from songshan_h3.compiler_compat import configure_hygon_compiler


def arrangement(value, blocks, output, *, tile=1024):
    return value.tile((1, 7, 128)), blocks, output.tile((tile,))


def application(value, blocks, output):
    index = output.offsets(0).to(ntl.int32)
    head = index // (blocks * 4096)
    tile = (index // 4096) % blocks
    local = index % 4096
    column = local // 32
    pair = (local // 2) % 16
    row = 2 * (pair ^ (((column // 2) % 8) * 2)) + local % 2
    output = value.source[tile * 32 + row, head, column]


@cache
def kernel():
    configure_hygon_compiler()
    return nt.make(
        arrangement, application,
        (nt.Tensor(shape=(None, 7, 128), dtype=nt.bfloat16),
         nt.Tensor(0, dtype=nt.int32), nt.Tensor(1, dtype=nt.bfloat16)),
        backend="triton", platform="hygon-bw1000",
        num_warps=4, num_stages=1, max_num_configs=1,
        kernel_name="h3_prepack_value",
    )
