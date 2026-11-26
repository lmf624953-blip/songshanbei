"""探索把 5376 维 RMSNorm 与按 token 选择的 AdaLN 调制合为一趟九齿计算。"""
import ninetoothed.language as ntl

from wide_norm import arrange as arrange_wide


def arrange_naive(hidden, weight, scale, shift, indices, output):
    hidden, weight, _, output = arrange_wide(hidden, weight, None, output)
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
