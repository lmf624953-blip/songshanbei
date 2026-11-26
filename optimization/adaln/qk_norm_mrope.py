"""九齿 Q/K RMSNorm 与三轴 MRoPE 融合原型；保留归约及 BF16 中间舍入。"""
import ninetoothed.language as ntl


def arrange(query, key, query_weight, key_weight, cos, sin, positions, query_out, key_out, *, rows=1):
    def tile(tensor):
        return tensor.tile((rows, 1, 128))
    return (tile(query), tile(key), query_weight, key_weight, cos, sin,
            positions, tile(query_out), tile(key_out))


def application(query, key, query_weight, key_weight, cos, sin, positions, query_out, key_out):
    row = query_out.offsets(0).to(ntl.int32)
    head = query_out.offsets(1).to(ntl.int32)
    dim = query_out.offsets(2).to(ntl.int32)
    # 与已验收的 cub_order_rmsnorm 相同，仅翻转归约索引低四位。
    permuted = ((dim & 1) << 3) + ((dim & 2) << 1)
    permuted += ((dim & 4) >> 1) + ((dim & 8) >> 3) + (dim & 112)
    q_red = query.source[row, head, permuted].to(ntl.float32)
    k_red = key.source[row, head, permuted].to(ntl.float32)
    q_squared = q_red * q_red + ntl.zeros(query.shape, dtype=ntl.float32)
    k_squared = k_red * k_red + ntl.zeros(key.shape, dtype=ntl.float32)
    q_scale = ntl.rsqrt(ntl.sum(q_squared, -1) / 128 + 1e-5)[:, :, None]
    k_scale = ntl.rsqrt(ntl.sum(k_squared, -1) / 128 + 1e-5)[:, :, None]
    q = ((query.to(ntl.float32) * query_weight.source[dim].to(ntl.float32)) * q_scale).to(query.dtype).to(ntl.float32)
    k = ((key.to(ntl.float32) * key_weight.source[dim].to(ntl.float32)) * k_scale).to(key.dtype).to(ntl.float32)
    partner = ntl.where(dim < 48, dim + 48, dim - 48)
    q_partner = ((query.source[row, head, partner].to(ntl.float32)
                  * query_weight.source[partner].to(ntl.float32)) * q_scale).to(query.dtype).to(ntl.float32)
    k_partner = ((key.source[row, head, partner].to(ntl.float32)
                  * key_weight.source[partner].to(ntl.float32)) * k_scale).to(key.dtype).to(ntl.float32)
    pair = dim % 48
    axis = pair // 16
    raw_position = positions.source[axis, row]
    position = ntl.where((raw_position >= 0) & (raw_position < cos.source.shape[0]), raw_position, 0).to(ntl.int32)
    cosine = cos.source[position, pair].to(ntl.float32)
    sine = sin.source[position, pair].to(ntl.float32)
    q_rotated = ntl.where(dim < 48, q * cosine - q_partner * sine, q * cosine + q_partner * sine)
    k_rotated = ntl.where(dim < 48, k * cosine - k_partner * sine, k * cosine + k_partner * sine)
    query_out = ntl.where(dim < 96, q_rotated, q)
    key_out = ntl.where(dim < 96, k_rotated, k)
