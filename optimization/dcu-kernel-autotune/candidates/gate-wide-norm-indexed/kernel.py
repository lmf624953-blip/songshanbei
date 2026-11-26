"""探索合并门控残差、宽 RMSNorm 和下一次索引 AdaLN，保留残差输出。"""
import ninetoothed.language as ntl


def arrange(hidden, update, weight, gate, scale, shift, indices, residual, output):
    h = hidden.tile((1, 8192))
    u = update.tile((1, 8192))
    r = residual.tile((1, 8192))
    out = output.tile((1, 8192))
    w = weight.tile((8192,)).unsqueeze(0).expand(out.shape)
    return h, u, w, gate, scale, shift, indices, r, out


def application(hidden, update, weight, gate, scale, shift, indices, residual, output):
    token = output.offsets(0).to(ntl.int32)
    column = output.offsets(1).to(ntl.int32)
    table_row = indices.source[token].to(ntl.int32)
    gated = (update.to(ntl.float32) * gate.source[table_row, column].to(ntl.float32)).to(output.dtype)
    summed = (hidden.to(ntl.float32) + gated.to(ntl.float32)).to(output.dtype)
    residual = summed
    x = summed.to(ntl.float32)
    variance = ntl.sum(x * x, -1) / 5376
    inv_rms = ntl.rsqrt(variance + 1e-5)[:, None]
    normalized = ((x * weight.to(ntl.float32)) * inv_rms).to(output.dtype).to(ntl.float32)
    scaled = (normalized * scale.source[table_row, column].to(ntl.float32)).to(output.dtype)
    shifted_input = (normalized + scaled.to(ntl.float32)).to(output.dtype)
    output = shifted_input.to(ntl.float32) + shift.source[table_row, column].to(ntl.float32)
