"""NineToothed attention retaining FP32 scale after the BF16 QK dot product."""
import ninetoothed.language as ntl
from functools import cache

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
            lse = lse * alpha + ntl.sum(probabilities, 1)
            acc = acc * alpha[:, None] + ntl.dot(probabilities.to(value[j].dtype), value[j])
            maximum = next_max
        for j in range(h3_full_blocks, key.shape[0]):
            qk = ntl.dot(query_i, ntl.trans(key[j])) * log2_scale
            qk = ntl.where(key[j].offsets(-2) < key.source.shape[-2], qk, float('-inf'))
            if is_causal:
                qk = ntl.where(query[i].offsets(-2)[:, None] + key.source.shape[-2] - query.source.shape[-2] >= key[j].offsets(-2)[None, :], qk, float('-inf'))
            next_max = ntl.maximum(maximum, ntl.max(qk, 1))
            probabilities = ntl.exp2(qk - next_max[:, None])
            alpha = ntl.exp2(maximum - next_max)
            lse = lse * alpha + ntl.sum(probabilities, 1)
            acc = acc * alpha[:, None] + ntl.dot(probabilities.to(value[j].dtype), value[j])
            maximum = next_max
        acc /= lse[:, None]
        output[i] = acc

@cache
def kernel():
    import ninetoothed as nt
    from songshan_h3.attention_kernel import definition
    from songshan_h3.compiler_compat import configure_hygon_compiler
    configure_hygon_compiler()
    arrangement, _, params = definition()
    return nt.make(arrangement, application, params, backend='triton', platform='hygon-bw1000', num_warps=4, num_stages=1, max_num_configs=1, kernel_name='h3_fp32_scale')
