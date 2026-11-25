"""NineToothed MiniMax-H3 AdaLN with the original BF16 rounding points."""

from functools import cache

import ninetoothed
import ninetoothed.language as ntl
import torch

from songshan_h3.compiler_compat import configure_hygon_compiler


def arrangement(normalized, scale, shift, output):
    return tuple(tensor.tile((1, 512)) for tensor in (normalized, scale, shift, output))


def adaln(normalized, scale, shift, output):
    scaled = (normalized.to(ntl.float32) * scale.to(ntl.float32)).to(normalized.dtype)
    residual = (normalized.to(ntl.float32) + scaled.to(ntl.float32)).to(normalized.dtype)
    output = (residual.to(ntl.float32) + shift.to(ntl.float32)).to(normalized.dtype)


@cache
def kernel():
    configure_hygon_compiler()
    return ninetoothed.make(
        arrangement, adaln,
        tuple(ninetoothed.Tensor(2, dtype=ninetoothed.bfloat16) for _ in range(4)),
        backend="triton", platform="hygon-bw1000", num_warps=4, num_stages=1,
        max_num_configs=1,
    )


def launch(out, normalized, scale, shift, stream):
    device = normalized.device
    stream = torch.cuda.ExternalStream(stream, device=device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        kernel()(normalized.squeeze(0), scale.squeeze(0), shift.squeeze(0), out.squeeze(0))
