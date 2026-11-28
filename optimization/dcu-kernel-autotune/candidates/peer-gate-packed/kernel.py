"""两个相邻BF16共享rank选择，保留两个元素各自的逐步RNE。"""

import ninetoothed.language as ntl


def arrangement(p0, p1, p2, p3, p4, p5, p6, p7, owners, residual, gate, indices,
                offset, ringbits, output, *, tile):
    return (p0, p1, p2, p3, p4, p5, p6, p7, owners, residual, gate, indices,
            offset, ringbits, output.tile((tile,)))


def application(p0, p1, p2, p3, p4, p5, p6, p7, owners, residual, gate, indices,
                offset, ringbits, output):
    local = output.offsets(0).to(ntl.int32)
    index = offset + local
    owner = owners.source[index // 4].to(ntl.int32)
    v0 = p0.source[index]
    v1 = p1.source[index]
    v2 = p2.source[index]
    v3 = p3.source[index]
    v4 = p4.source[index]
    v5 = p5.source[index]
    v6 = p6.source[index]
    v7 = p7.source[index]
    r = ((ringbits >> (3 * ((owner + 1) & 7))) & 7).to(ntl.int32)
    acc = ntl.where(r == 0, v0, ntl.where(r == 1, v1, ntl.where(r == 2, v2,
          ntl.where(r == 3, v3, ntl.where(r == 4, v4, ntl.where(r == 5, v5,
          ntl.where(r == 6, v6, v7))))))).to(ntl.uint32)
    for step in range(2, 9):
        r = ((ringbits >> (3 * ((owner + step) & 7))) & 7).to(ntl.int32)
        value = ntl.where(r == 0, v0, ntl.where(r == 1, v1, ntl.where(r == 2, v2,
                ntl.where(r == 3, v3, ntl.where(r == 4, v4, ntl.where(r == 5, v5,
                ntl.where(r == 6, v6, v7))))))).to(ntl.uint32)
        low_a = (acc << 16).to(ntl.uint32).to(ntl.float32, bitcast=True)
        high_a = (acc & 0xffff0000).to(ntl.uint32).to(ntl.float32, bitcast=True)
        low_b = (value << 16).to(ntl.uint32).to(ntl.float32, bitcast=True)
        high_b = (value & 0xffff0000).to(ntl.uint32).to(ntl.float32, bitcast=True)
        low_bits = (low_a + low_b).to(ntl.uint32, bitcast=True)
        high_bits = (high_a + high_b).to(ntl.uint32, bitcast=True)
        low_round = (low_bits + 0x7fff + ((low_bits >> 16) & 1)).to(ntl.uint32) >> 16
        high_round = (high_bits + 0x7fff + ((high_bits >> 16) & 1)) & 0xffff0000
        acc = (low_round | high_round).to(ntl.uint32)
    row = index // 2688
    column = index % 2688
    g = gate.source[indices.source[row].to(ntl.int32), column]
    res = residual.source[index]
    low_acc = (acc << 16).to(ntl.uint32).to(ntl.float32, bitcast=True)
    high_acc = (acc & 0xffff0000).to(ntl.uint32).to(ntl.float32, bitcast=True)
    low_gate = (g << 16).to(ntl.uint32).to(ntl.float32, bitcast=True)
    high_gate = (g & 0xffff0000).to(ntl.uint32).to(ntl.float32, bitcast=True)
    low_res = (res << 16).to(ntl.uint32).to(ntl.float32, bitcast=True)
    high_res = (res & 0xffff0000).to(ntl.uint32).to(ntl.float32, bitcast=True)
    low_scaled = (low_gate * low_acc).to(ntl.bfloat16).to(ntl.float32)
    high_scaled = (high_gate * high_acc).to(ntl.bfloat16).to(ntl.float32)
    low_output = (low_res + low_scaled).to(ntl.bfloat16).to(ntl.float32)
    high_output = (high_res + high_scaled).to(ntl.bfloat16).to(ntl.float32)
    low_result = low_output.to(ntl.uint32, bitcast=True) >> 16
    high_result = high_output.to(ntl.uint32, bitcast=True) & 0xffff0000
    output = (low_result | high_result).to(ntl.uint32)
