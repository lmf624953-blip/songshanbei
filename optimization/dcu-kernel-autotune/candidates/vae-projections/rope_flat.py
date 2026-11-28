"""用一维偏移计算 VAE RoPE 的通道及旋转配对。"""

from functools import partial

import ninetoothed as nt
import ninetoothed.language as ntl
from songshan_h3.compiler_compat import configure_hygon_compiler


def arrange(q, k, cosine, sine, output_q, output_k, *, tile):
    return (
        q.tile((tile,)), k.tile((tile,)), cosine, sine,
        output_q.tile((tile,)), output_k.tile((tile,)),
    )


def apply_rope_flat(q, k, cosine, sine, output_q, output_k):
    index = output_q.offsets(0).to(ntl.int32)
    row = index // 2048
    dim = index % 64
    col = ntl.where(dim < 48, dim, 0)
    partner = index - dim + ntl.where(col < 24, col + 24, col - 24)
    c = cosine.source[row, col].to(ntl.float32)
    s = sine.source[row, col].to(ntl.float32)
    q_value = q.to(ntl.float32)
    k_value = k.to(ntl.float32)
    q_rotate = q.source[partner].to(ntl.float32)
    k_rotate = k.source[partner].to(ntl.float32)
    q_rotate = ntl.where(col < 24, -q_rotate, q_rotate)
    k_rotate = ntl.where(col < 24, -k_rotate, k_rotate)
    q_cos = (q_value * c).to(ntl.bfloat16).to(ntl.float32)
    k_cos = (k_value * c).to(ntl.bfloat16).to(ntl.float32)
    q_sin = (q_rotate * s).to(ntl.bfloat16).to(ntl.float32)
    k_sin = (k_rotate * s).to(ntl.bfloat16).to(ntl.float32)
    output_q = ntl.where(dim < 48, q_cos + q_sin, q_value)
    output_k = ntl.where(dim < 48, k_cos + k_sin, k_value)


def build_rope_flat(tile):
    configure_hygon_compiler()
    flat, frequency = (None,), (None, 48)
    shapes = flat, flat, frequency, frequency, flat, flat
    tensors = tuple(nt.Tensor(shape=shape, dtype=nt.bfloat16) for shape in shapes)
    return nt.make(partial(arrange, tile=tile), apply_rope_flat, tensors,
                   backend="triton", platform="hygon-bw1000", num_warps=4,
                   num_stages=1, max_num_configs=1)
