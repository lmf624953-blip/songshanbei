"""九齿三轴 MRoPE 原型：128 维 head、96 维旋转，保留原 BF16/FP32 运算。"""
import ninetoothed.language as ntl


def arrange(query, key, cos, sin, positions, query_out, key_out, *, rows=4):
    def tile(tensor):
        return tensor.tile((rows, 1, 128))
    return tile(query), tile(key), cos, sin, positions, tile(query_out), tile(key_out)


def application(query, key, cos, sin, positions, query_out, key_out):
    row = query_out.offsets(0).to(ntl.int32)
    head = query_out.offsets(1).to(ntl.int32)
    dim = query_out.offsets(2).to(ntl.int32)
    pair = dim % 48
    axis = pair // 16
    raw_position = positions.source[axis, row]
    position = ntl.where((raw_position >= 0) & (raw_position < cos.source.shape[0]), raw_position, 0).to(ntl.int32)
    cosine = cos.source[position, pair].to(ntl.float32)
    sine = sin.source[position, pair].to(ntl.float32)
    partner = ntl.where(dim < 48, dim + 48, dim - 48)
    q = query.to(ntl.float32)
    k = key.to(ntl.float32)
    q_partner = query.source[row, head, partner].to(ntl.float32)
    k_partner = key.source[row, head, partner].to(ntl.float32)
    q_rotated = ntl.where(dim < 48, q * cosine - q_partner * sine, q * cosine + q_partner * sine)
    k_rotated = ntl.where(dim < 48, k * cosine - k_partner * sine, k * cosine + k_partner * sine)
    query_out = ntl.where(dim < 96, q_rotated, q)
    key_out = ntl.where(dim < 96, k_rotated, k)
