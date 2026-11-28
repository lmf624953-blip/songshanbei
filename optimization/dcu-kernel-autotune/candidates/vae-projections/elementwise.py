"""VAE RoPE 与 SwiGLU 九齿原型，显式保留原有的 BF16 舍入。"""

from functools import partial

import ninetoothed as nt
import ninetoothed.language as ntl
from songshan_h3.compiler_compat import configure_hygon_compiler


def arrange_rope(q, k, cosine, sine, output_q, output_k, *, heads):
    return q.tile((1, heads, 64)), k.tile((1, heads, 64)), cosine, sine, output_q.tile(
        (1, heads, 64)), output_k.tile((1, heads, 64))


def apply_rope(q, k, cosine, sine, output_q, output_k):
    row = output_q.offsets(0).to(ntl.int32)
    head = output_q.offsets(1).to(ntl.int32)
    dim = output_q.offsets(2).to(ntl.int32)
    col = ntl.where(dim < 48, dim, 0)
    partner = ntl.where(col < 24, col + 24, col - 24)
    c = cosine.source[row, col].to(ntl.float32)
    s = sine.source[row, col].to(ntl.float32)
    q_other = q.source[row, head, partner].to(ntl.float32)
    k_other = k.source[row, head, partner].to(ntl.float32)
    q_other = ntl.where(col < 24, -q_other, q_other)
    k_other = ntl.where(col < 24, -k_other, k_other)
    q_cos = (q.to(ntl.float32) * c).to(ntl.bfloat16).to(ntl.float32)
    k_cos = (k.to(ntl.float32) * c).to(ntl.bfloat16).to(ntl.float32)
    q_sin = (q_other * s).to(ntl.bfloat16).to(ntl.float32)
    k_sin = (k_other * s).to(ntl.bfloat16).to(ntl.float32)
    output_q = ntl.where(dim < 48, q_cos + q_sin, q.to(ntl.float32))
    output_k = ntl.where(dim < 48, k_cos + k_sin, k.to(ntl.float32))


def arrange_swiglu(up, gate, output, *, tile):
    def tiled(tensor):
        result = tensor.tile((1, tile))
        result.dtype = result.dtype.squeeze(0)
        return result
    return tiled(up), tiled(gate), tiled(output)


def silu_div(up, gate, output):
    g = gate.to(ntl.float32)
    activation = (g / (1.0 + ntl.exp(-g))).to(ntl.bfloat16).to(ntl.float32)
    output = activation * up.to(ntl.float32)


def silu_fast(up, gate, output):
    g = gate.to(ntl.float32)
    activation = ntl.libdevice.fast_dividef(g, 1.0 + ntl.libdevice.fast_expf(-g))
    output = activation.to(ntl.bfloat16).to(ntl.float32) * up.to(ntl.float32)


def silu_mul(up, gate, output):
    g = gate.to(ntl.float32)
    activation = g * (1.0 / (1.0 + ntl.exp(-g)))
    output = activation.to(ntl.bfloat16).to(ntl.float32) * up.to(ntl.float32)


def build_rope(heads):
    configure_hygon_compiler()
    qkv_shape = (None, 32, 64)
    shapes = (qkv_shape, qkv_shape, (None, 48), (None, 48), qkv_shape, qkv_shape)
    tensors = tuple(nt.Tensor(shape=shape, dtype=nt.bfloat16) for shape in shapes)
    return nt.make(partial(arrange_rope, heads=heads), apply_rope, tensors,
                   backend="triton", platform="hygon-bw1000", num_warps=4,
                   num_stages=1, max_num_configs=1)


def build_swiglu(operation, tile=1024):
    configure_hygon_compiler()
    tensors = tuple(nt.Tensor(shape=(None, 8192), dtype=nt.bfloat16) for _ in range(3))
    return nt.make(partial(arrange_swiglu, tile=tile), operation, tensors,
                   backend="triton", platform="hygon-bw1000", num_warps=4,
                   num_stages=1, max_num_configs=1)
