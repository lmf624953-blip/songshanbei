"""九齿 K32 长注意力：完整 KV 块与带掩码尾块分开，保持原归约顺序。"""
import ninetoothed.language as ntl

def application(query, key, value, is_causal, scale, output):
    log2_scale = (1.4426950408889634 * scale).to(ntl.float32)
    for i in range(query.shape[0]):
        query_i = query[i]
        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        maximum = ntl.full((query_i.shape[-2],), float('-inf'), dtype=ntl.float32)
        h3_full_blocks = key.source.shape[-2] // 32
        for j in range(h3_full_blocks):
            qk = ntl.dot(query_i, ntl.trans(key[j])) * log2_scale
            if is_causal:
                qk = ntl.where(query[i].offsets(-2)[:, None] + key.source.shape[-2] - query.source.shape[-2] >= key[j].offsets(-2)[None, :], qk, float('-inf'))
            next_max = ntl.maximum(maximum, ntl.max(qk, 1))
            probabilities = ntl.exp2(qk - next_max[:, None])
            alpha = ntl.exp2(maximum - next_max)
            acc = acc * alpha[:, None] + ntl.dot(probabilities.to(ntl.float16), value[j].to(ntl.float16))
            maximum = next_max
            lse = lse * alpha + ntl.sum(probabilities, 1)
        for j in range(h3_full_blocks, key.shape[0]):
            qk = ntl.dot(query_i, ntl.trans(key[j])) * log2_scale
            qk = ntl.where(key[j].offsets(-2) < key.source.shape[-2], qk, float('-inf'))
            if is_causal:
                qk = ntl.where(query[i].offsets(-2)[:, None] + key.source.shape[-2] - query.source.shape[-2] >= key[j].offsets(-2)[None, :], qk, float('-inf'))
            next_max = ntl.maximum(maximum, ntl.max(qk, 1))
            probabilities = ntl.exp2(qk - next_max[:, None])
            alpha = ntl.exp2(maximum - next_max)
            acc = acc * alpha[:, None] + ntl.dot(probabilities.to(ntl.float16), value[j].to(ntl.float16))
            maximum = next_max
            lse = lse * alpha + ntl.sum(probabilities, 1)
        acc /= lse[:, None]
        output[i] = acc
