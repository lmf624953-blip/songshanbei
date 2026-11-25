"""NineToothed Q/K RMSNorm with FP32 arithmetic and BF16 output."""
import ninetoothed.language as ntl


def arrange(input, weight, epsilon, output, *, rows, heads, width):
    def tile(tensor):
        return tensor.tile((rows, heads, width))

    result = tile(output)
    weight = weight.tile((width,)).unsqueeze(0).unsqueeze(0).expand(result.shape)
    return tile(input), weight, epsilon, result


def rmsnorm(input, weight, epsilon, output):
    x = input.to(ntl.float32)
    variance = ntl.sum(x * x, -1) / input.shape[-1]
    rsqrt = ntl.rsqrt(variance + epsilon)
    output = (x * weight.to(ntl.float32)) * rsqrt[:, :, None]


def adjacent_rmsnorm(input, weight, epsilon, output):
    x = input.to(ntl.float32)
    row = input.offsets(0).to(ntl.int32)
    head = input.offsets(1).to(ntl.int32)
    d = input.offsets(2).to(ntl.int32)
    # The InfiniCore warp reduction combines neighbors before increasing the offset.
    permuted = ((d & 1) << 5) + ((d & 2) << 3) + ((d & 4) << 1)
    permuted += ((d & 8) >> 1) + ((d & 16) >> 3) + ((d & 32) >> 5) + (d & 64)
    red = input.source[row, head, permuted].to(ntl.float32)
    squares = red * red + ntl.zeros(input.shape, dtype=ntl.float32)
    variance = ntl.sum(squares, -1) / input.shape[-1]
    rsqrt = ntl.rsqrt(variance + epsilon)
    output = (x * weight.to(ntl.float32)) * rsqrt[:, :, None]


def cub_order_rmsnorm(input, weight, epsilon, output):
    x = input.to(ntl.float32)
    row = input.offsets(0).to(ntl.int32)
    head = input.offsets(1).to(ntl.int32)
    d = input.offsets(2).to(ntl.int32)
    # The emitted Triton reduction combines offsets 8,4,2,1,16,32.
    # CUB combines 1,2,4,8,16,32. Reverse only the low four bits.
    permuted = ((d & 1) << 3) + ((d & 2) << 1)
    permuted += ((d & 4) >> 1) + ((d & 8) >> 3) + (d & 112)
    red = input.source[row, head, permuted].to(ntl.float32)
    squares = red * red + ntl.zeros(input.shape, dtype=ntl.float32)
    variance = ntl.sum(squares, -1) / input.shape[-1]
    rsqrt = ntl.rsqrt(variance + epsilon)
    output = (x * weight.to(ntl.float32)) * rsqrt[:, :, None]


