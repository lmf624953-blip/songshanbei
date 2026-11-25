"""NineToothed fusions that read each token's AdaLN modulation directly."""
from functools import cache

import ninetoothed as nt
import ninetoothed.language as ntl

from songshan_h3.compiler_compat import configure_hygon_compiler


def activation_tiles(tensor, *, tile=256):
    tiled = tensor.tile((1, tile))
    tiled.dtype = tiled.dtype.squeeze(0)
    return tiled


def index_tiles(indices, out):
    tiled = indices.tile((1,)).unsqueeze(1).expand((-1, out.shape[1]))
    tiled.dtype = tiled.dtype.squeeze(0)
    return tiled


def arrange_adaln(normalized, scale, shift, indices, output, *, tile=256):
    out = activation_tiles(output, tile=tile)
    return activation_tiles(normalized, tile=tile), scale, shift, index_tiles(indices, out), out


def indexed_adaln(normalized, scale, shift, indices, output):
    row = indices.to(ntl.int32)
    column = output.offsets(1)
    scale_value = scale.source[row, column]
    shift_value = shift.source[row, column]
    scaled = (normalized.to(ntl.float32) * scale_value.to(ntl.float32)).to(normalized.dtype)
    residual = (normalized.to(ntl.float32) + scaled.to(ntl.float32)).to(normalized.dtype)
    output = (residual.to(ntl.float32) + shift_value.to(ntl.float32)).to(normalized.dtype)


def arrange_gate(residual, activation, gate, indices, output, *, tile=256):
    out = activation_tiles(output, tile=tile)
    return activation_tiles(residual, tile=tile), activation_tiles(activation, tile=tile), gate, index_tiles(indices, out), out


def indexed_gate(residual, activation, gate, indices, output):
    gate_value = gate.source[indices.to(ntl.int32), output.offsets(1)]
    scaled = (gate_value.to(ntl.float32) * activation.to(ntl.float32)).to(activation.dtype)
    output = (residual.to(ntl.float32) + scaled.to(ntl.float32)).to(residual.dtype)


@cache
def kernels():
    configure_hygon_compiler()
    tensors = (
        nt.Tensor(2, dtype=nt.bfloat16), nt.Tensor(2, dtype=nt.bfloat16),
        nt.Tensor(2, dtype=nt.bfloat16), nt.Tensor(1, dtype=nt.int64),
        nt.Tensor(2, dtype=nt.bfloat16),
    )
    return tuple(
        nt.make(arrangement, application, tensors, backend="triton", platform="hygon-bw1000",
                num_warps=4, num_stages=1, max_num_configs=1)
        for arrangement, application in ((arrange_adaln, indexed_adaln), (arrange_gate, indexed_gate))
    )
