"""NineToothed online softmax attention for the H3 dense attention path.

The arrangement adapts the pinned ntops source, padding the head tile to 128
for the vision encoder's 72-wide heads. The application avoids
branch-local mask definitions that the current NineToothed SSA frontend rejects.
No generated Triton or CUDA source is edited here.
"""
from functools import cache

import ninetoothed
import ninetoothed.language as ntl


def arrangement(query, key, value, is_causal, scale, output):
    def query_or_output(tensor):
        arranged = tensor.tile((1, 1, 64, 128)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1))
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    query_arranged = query_or_output(query)

    def key_or_value(tensor):
        arranged = (tensor.tile((1, 1, 64, 128)).tile((1, 1, -1, -1))
                    .expand((-1, -1, query_arranged.shape[-2], -1)))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    return (query_arranged, key_or_value(key), key_or_value(value),
            is_causal, scale, query_or_output(output))


def application(query, key, value, is_causal, scale, output):
    for i in range(query.shape[0]):
        query_i = (1.4426950408889634 * scale * query[i]).to(query[i].dtype)
        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        maximum = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)
        for j in range(key.shape[0]):
            qk = ntl.dot(query_i, ntl.trans(key[j]))
            qk = ntl.where(key[j].offsets(-2) < key.source.shape[-2], qk, float("-inf"))
            if is_causal:
                # InfiniCore's causal varlen ABI uses lower-right alignment.
                qk = ntl.where(
                    query[i].offsets(-2)[:, None] + key.source.shape[-2]
                    - query.source.shape[-2] >= key[j].offsets(-2)[None, :],
                    qk, float("-inf"))
            next_max = ntl.maximum(maximum, ntl.max(qk, 1))
            probabilities = ntl.exp2(qk - next_max[:, None])
            alpha = ntl.exp2(maximum - next_max)
            acc = acc * alpha[:, None] + ntl.dot(probabilities.to(value[j].dtype), value[j])
            maximum = next_max
            lse = lse * alpha + ntl.sum(probabilities, 1)
        acc /= lse[:, None]
        output[i] = acc


def definition(head_dim=None, dtype=None, causal=None):
    query, key, value, output = (
        ninetoothed.Tensor(
            4, dtype=dtype,
            shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128}))
        for _ in range(4))
    if head_dim is not None:
        for tensor in (query, key, value, output):
            tensor.shape = tensor.shape[:-1] + (head_dim,)
    is_causal = ninetoothed.Tensor(0, constexpr=True, value=causal)
    scale = ninetoothed.Tensor(0, dtype=ninetoothed.float64)
    return arrangement, application, (query, key, value, is_causal, scale, output)


@cache
def kernel():
    from .compiler_compat import configure_hygon_compiler
    configure_hygon_compiler()
    return ninetoothed.make(
        *definition(), backend="triton", platform="hygon-bw1000",
        num_warps=4, num_stages=1, max_num_configs=1)
