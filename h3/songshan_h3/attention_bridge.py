"""InfiniCore host dispatcher to NineToothed; no ATen numerical fallback."""
import torch

from .attention_kernel import kernel


def _validate(out, q, k, v):
    if any(t.ndim != 4 for t in (out, q, k, v)):
        raise ValueError('Attention expects [batch, sequence, heads, dimension]')
    if not q.is_cuda or any(t.device != q.device for t in (out, k, v)):
        raise ValueError('Attention tensors must share a Hygon device')
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('H3 attention supports FP16 and BF16')
    if any(t.dtype != q.dtype for t in (out, k, v)):
        raise ValueError('Attention tensor dtypes must match')
    if not (q.shape[0] == k.shape[0] == v.shape[0] and k.shape[1:3] == v.shape[1:3]
            and q.shape[3] == k.shape[3] == v.shape[3] and tuple(out.shape) == tuple(q.shape)):
        raise ValueError('Inconsistent attention tensor shapes')
    if q.shape[3] not in (16, 32, 64, 72, 128) or q.shape[2] % k.shape[2]:
        raise ValueError('Unsupported head dimension or head group ratio')
    if not q.shape[1] or not k.shape[1]:
        raise ValueError('Empty attention sequences are unsupported')


def _launch(out, q, k, v, scale, causal):
    _validate(out, q, k, v)
    q, k, v, out = (t.permute(0, 2, 1, 3) for t in (q, k, v, out))
    with torch.cuda.device(q.device):
        kernel()(q, k, v, causal, scale, out)


def dense(out, q, k, v, scale, causal, stream):
    with torch.cuda.device(q.device), torch.cuda.stream(torch.cuda.ExternalStream(stream, device=q.device)):
        _launch(out, q, k, v, scale, causal)


def varlen(out, q, k, v, cu_q, cu_k, scale, stream):
    if any(t.ndim != 3 for t in (out, q, k, v)):
        raise ValueError('Varlen attention expects packed [sequence, heads, dimension]')
    with torch.cuda.device(q.device), torch.cuda.stream(torch.cuda.ExternalStream(stream, device=q.device)):
        # Prefix lengths are host control metadata, not model activations.
        offsets_q, offsets_k = cu_q.tolist(), cu_k.tolist()
        if (len(offsets_q) != len(offsets_k) or len(offsets_q) < 2
                or offsets_q[0] != 0 or offsets_k[0] != 0
                or offsets_q[-1] != q.shape[0] or offsets_k[-1] != k.shape[0]
                or any(a >= b for a, b in zip(offsets_q, offsets_q[1:]))
                or any(a >= b for a, b in zip(offsets_k, offsets_k[1:]))):
            raise ValueError('Invalid cumulative sequence lengths')
        for a, b, c, d in zip(offsets_q, offsets_q[1:], offsets_k, offsets_k[1:]):
            if b - a > d - c:
                raise ValueError('Causal attention requires query length <= key length')
            _launch(out[a:b].unsqueeze(0), q[a:b].unsqueeze(0),
                    k[c:d].unsqueeze(0), v[c:d].unsqueeze(0), scale, True)


def install():
    import infinicore  # Load the original dispatcher shared library first.
    from . import _attention_bridge
    kernel()  # Materialize the JIT wrapper before InferEngine worker threads start.
    if torch.cuda.is_available():
        # Triton's HIP driver is lazily initialized without a Python lock.
        # Concurrent first calls can deadlock dl_iterate_phdr's loader lock
        # against the GIL while another worker imports hip_utils. Initialize
        # the backend on the calling thread before any native workers exist.
        from triton.runtime import driver
        driver.active.get_current_target()
    _attention_bridge.register_hygon()
