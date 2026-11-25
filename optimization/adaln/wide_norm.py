"""九齿宽 RMSNorm 实验；不参与当前选定配置。"""
import ninetoothed.language as ntl


def arrange(input, weight, epsilon, output):
    x = input.tile((1, 8192))
    out = output.tile((1, 8192))
    w = weight.tile((8192,)).unsqueeze(0).expand(out.shape)
    return x, w, epsilon, out


def naive(input, weight, epsilon, output):
    x = input.to(ntl.float32)
    variance = ntl.sum(x * x, -1) / 5376
    output = (x * weight.to(ntl.float32)) * ntl.rsqrt(variance + epsilon)[:, None]


def arrange_chunked(input, weight, epsilon, output):
    x = input.tile((1, 1024)).tile((1, -1))
    x.dtype = x.dtype.squeeze(0)
    x.dtype.dtype = x.dtype.dtype.squeeze(0)
    out = output.tile((1, 8192))
    w = weight.tile((8192,)).unsqueeze(0).expand(out.shape)
    return x, w, epsilon, out


def chunked(input, weight, epsilon, output):
    total = ntl.zeros((1024,), dtype=ntl.float32)
    for part in range(input.shape[0]):
        value = input[part].to(ntl.float32)
        total = total + value * value
    variance = ntl.sum(total, 0) / input.source.shape[1]
    x = input.source[output.offsets(0), output.offsets(1)].to(ntl.float32)
    output = (x * weight.to(ntl.float32)) * ntl.rsqrt(variance + epsilon)


def cub_order(input, weight, epsilon, output):
    total = ntl.zeros((1024,), dtype=ntl.float32)
    for part in range(input.shape[0]):
        row = input[part].offsets(0).to(ntl.int32)
        d = input[part].offsets(1).to(ntl.int32)
        # Preserve the chunk offset and reverse only the low four bits.
        permuted = ((d & 1) << 3) + ((d & 2) << 1) + ((d & 4) >> 1) + ((d & 8) >> 3) + (d & -16)
        value = input.source[row, permuted].to(ntl.float32)
        total = total + value * value
    variance = ntl.sum(total, 0) / input.source.shape[1]
    x = input.source[output.offsets(0), output.offsets(1)].to(ntl.float32)
    output = (x * weight.to(ntl.float32)) * ntl.rsqrt(variance + epsilon)
