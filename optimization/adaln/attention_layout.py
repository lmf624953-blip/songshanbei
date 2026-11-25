"""九齿注意力的可配置行分块；保持相同的头布局和完整 KV 遍历。"""


def arrangement(query, key, value, is_causal, scale, output, *, query_block=64, key_block=64):
    def query_or_output(tensor):
        arranged = tensor.tile((1, 1, query_block, 128)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1))
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    query_arranged = query_or_output(query)

    def key_or_value(tensor):
        arranged = tensor.tile((1, 1, key_block, 128)).tile((1, 1, -1, -1)).expand(
            (-1, -1, query_arranged.shape[-2], -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    return (query_arranged, key_or_value(key), key_or_value(value),
            is_causal, scale, query_or_output(output))
