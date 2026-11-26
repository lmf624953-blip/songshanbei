"""探索把 5376 维 RMSNorm 与按 token 选择的 AdaLN 调制合为一趟九齿计算。"""
import ninetoothed.language as ntl

from wide_norm import arrange as arrange_wide, arrange_chunked


def arrange_naive(hidden, weight, scale, shift, indices, output):
    hidden, weight, _, output = arrange_wide(hidden, weight, None, output)
    return hidden, weight, scale, shift, indices, output


def arrange_cub(hidden, weight, scale, shift, indices, output):
    hidden, weight, _, output = arrange_chunked(hidden, weight, None, output)
    return hidden, weight, scale, shift, indices, output


def naive(hidden, weight, scale, shift, indices, output):
    token = output.offsets(0).to(ntl.int32)
    column = output.offsets(1).to(ntl.int32)
    table_row = indices.source[token].to(ntl.int32)
    x = hidden.to(ntl.float32)
    variance = ntl.sum(x * x, -1) / 5376
    inv_rms = ntl.rsqrt(variance + 1e-5)[:, None]
    normalized = ((x * weight.to(ntl.float32)) * inv_rms).to(output.dtype).to(ntl.float32)
    scaled = (normalized * scale.source[table_row, column].to(ntl.float32)).to(output.dtype)
    residual = (normalized + scaled.to(ntl.float32)).to(output.dtype)
    output = residual.to(ntl.float32) + shift.source[table_row, column].to(ntl.float32)


def cub(hidden, weight, scale, shift, indices, output):
    partial = ntl.zeros((1024,), dtype=ntl.float32)
    for part in range(hidden.shape[0]):
        token = hidden[part].offsets(0).to(ntl.int32)
        d = hidden[part].offsets(1).to(ntl.int32)
        permuted = ((d & 1) << 3) + ((d & 2) << 1) + ((d & 4) >> 1) + ((d & 8) >> 3) + (d & -16)
        value = hidden.source[token, permuted].to(ntl.float32)
        partial = partial + value * value
    inv_rms = ntl.rsqrt(ntl.sum(partial, 0) / 5376 + 1e-5)
    token = output.offsets(0).to(ntl.int32)
    column = output.offsets(1).to(ntl.int32)
    table_row = indices.source[token].to(ntl.int32)
    x = hidden.source[token, column].to(ntl.float32)
    normalized = ((x * weight.to(ntl.float32)) * inv_rms).to(output.dtype).to(ntl.float32)
    scaled = (normalized * scale.source[table_row, column].to(ntl.float32)).to(output.dtype)
    residual = (normalized + scaled.to(ntl.float32)).to(output.dtype)
    output = residual.to(ntl.float32) + shift.source[table_row, column].to(ntl.float32)


def cub_vector(hidden, weight, scale, shift, indices, output):
    token = output.offsets(0).to(ntl.int32)
    column = output.offsets(1).to(ntl.int32)
    d = ((column & 1) << 3) + ((column & 2) << 1) + ((column & 4) >> 1)
    d = d + ((column & 8) >> 3) + (column & -16)
    d = ntl.where(column < 1024, d, -8192)
    x0 = hidden.source[token, d].to(ntl.float32)
    x1 = hidden.source[token, d + 1024].to(ntl.float32)
    x2 = hidden.source[token, d + 2048].to(ntl.float32)
    x3 = hidden.source[token, d + 3072].to(ntl.float32)
    x4 = hidden.source[token, d + 4096].to(ntl.float32)
    x5 = hidden.source[token, d + 5120].to(ntl.float32)
    partial = (((x0 * x0 + x1 * x1) + x2 * x2) + x3 * x3) + x4 * x4
    partial = partial + x5 * x5 + ntl.zeros(output.shape, dtype=ntl.float32)
    inv_rms = ntl.rsqrt(ntl.sum(partial, -1) / 5376 + 1e-5)[:, None]
    table_row = indices.source[token].to(ntl.int32)
    x = hidden.to(ntl.float32)
    normalized = ((x * weight.to(ntl.float32)) * inv_rms).to(output.dtype).to(ntl.float32)
    scaled = (normalized * scale.source[table_row, column].to(ntl.float32)).to(output.dtype)
    residual = (normalized + scaled.to(ntl.float32)).to(output.dtype)
    output = residual.to(ntl.float32) + shift.source[table_row, column].to(ntl.float32)


def cub_padded(hidden, weight, scale, shift, indices, output):
    token = output.offsets(0).to(ntl.int32)
    column = output.offsets(1).to(ntl.int32)
    d = ((column & 1) << 3) + ((column & 2) << 1) + ((column & 4) >> 1)
    d = d + ((column & 8) >> 3) + (column & 1008)
    # 每次读取使用有效列号，避免九齿把末段的加载掩码传播到前五段的加和。
    x0 = hidden.source[token, d].to(ntl.float32)
    x1 = hidden.source[token, d + 1024].to(ntl.float32)
    x2 = hidden.source[token, d + 2048].to(ntl.float32)
    x3 = hidden.source[token, d + 3072].to(ntl.float32)
    x4 = hidden.source[token, d + 4096].to(ntl.float32)
    x5 = hidden.source[token, (d & 255) + 5120].to(ntl.float32)
    partial = (((x0 * x0 + x1 * x1) + x2 * x2) + x3 * x3) + x4 * x4
    partial = partial + ntl.where(d < 256, x5 * x5, 0.0)
    partial = ntl.where(column < 1024, partial, 0.0) + ntl.zeros(output.shape, dtype=ntl.float32)
    inv_rms = ntl.rsqrt(ntl.sum(partial, -1) / 5376 + 1e-5)[:, None]
    table_row = indices.source[token].to(ntl.int32)
    x = hidden.to(ntl.float32)
    normalized = ((x * weight.to(ntl.float32)) * inv_rms).to(output.dtype).to(ntl.float32)
    scaled = (normalized * scale.source[table_row, column].to(ntl.float32)).to(output.dtype)
    residual = (normalized + scaled.to(ntl.float32)).to(output.dtype)
    output = residual.to(ntl.float32) + shift.source[table_row, column].to(ntl.float32)
