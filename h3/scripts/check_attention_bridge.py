#!/usr/bin/env python3
"""Check NineToothed dispatch numerically before loading the H3 weights."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def math_attention_reference(query, key, value, *, attn_mask=None,
                             is_causal=False, enable_gqa=False):
    """Small correctness oracle; never selects an external flash library."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask,
            is_causal=is_causal, enable_gqa=enable_gqa)


def check_device(device_id):
    import infinicore as ic
    import torch

    ic.set_device(ic.device('cuda', device_id))
    device = f'cuda:{device_id}'

    def wrap(tensor):
        # from_torch in the pinned Core branch discards strides.
        return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                    dtype=ic.bfloat16, device=ic.device('cuda', device_id))

    checks = []
    for nq, nk, hq, hk, dim, causal, strided in (
            (4, 4, 2, 2, 128, False, False), (67, 73, 4, 2, 128, False, True),
            (65, 65, 2, 2, 128, True, False), (7, 71, 4, 2, 128, True, True),
            (67, 67, 2, 2, 72, False, False)):
        def allocate(sequence, heads):
            tensor = torch.randn(1, sequence, heads, dim * (2 if strided else 1),
                                 device=device, dtype=torch.bfloat16)
            return tensor[..., ::2] if strided else tensor

        q, k, v = allocate(nq, hq), allocate(nk, hk), allocate(nk, hk)
        out = torch.empty_like(q)
        torch.cuda.synchronize(device_id)
        ic.mha(*(wrap(t) for t in (q, k, v)), scale=dim**-0.5, is_causal=causal, out=wrap(out))
        ic.sync_device()
        mask = (torch.arange(nq, device=device)[:, None] + nk - nq
                >= torch.arange(nk, device=device)[None, :]) if causal else None
        reference = math_attention_reference(
            *(t.permute(0, 2, 1, 3) for t in (q, k, v)),
            attn_mask=mask, enable_gqa=(hq != hk)).permute(0, 2, 1, 3)
        torch.testing.assert_close(out, reference, rtol=0.04, atol=0.03)
        checks.append(dict(kind='dense', nq=nq, nk=nk, q_heads=hq, kv_heads=hk,
                           head_dim=dim, causal=causal, strided=strided,
                           max_abs_error=float((out.float() - reference.float()).abs().max())))

    q, k, v = (torch.randn(12, 2, 128, device=device, dtype=torch.bfloat16) for _ in range(3))
    offsets = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    torch.cuda.synchronize(device_id)
    ic.mha_varlen(*(ic.from_torch(t) for t in (q, k, v, offsets, offsets)),
                  block_table=None, max_seqlen_q=7, max_seqlen_k=7,
                  scale=128**-0.5, out=ic.from_torch(out))
    ic.sync_device()
    for start, end in ((0, 5), (5, 12)):
        reference = math_attention_reference(
            *(t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v)),
            is_causal=True).squeeze(0).transpose(0, 1)
        torch.testing.assert_close(out[start:end], reference, rtol=0.04, atol=0.03)
    checks.append(dict(kind='varlen', sequence_lengths=[5, 7], causal=True))

    # The video VAE uses 64-wide heads and can batch multiple latent chunks.
    from songshan_h3.diffusers_attention import video_attention
    q, k, v = (torch.randn(2, 67, 4, 64, device=device, dtype=torch.bfloat16) for _ in range(3))
    out = video_attention(q, k, v)
    reference = math_attention_reference(
        *(t.permute(0, 2, 1, 3) for t in (q, k, v))).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, reference, rtol=0.04, atol=0.03)
    checks.append(dict(kind='video_vae', batch_size=2, head_dim=64,
                       max_abs_error=float((out.float() - reference.float()).abs().max())))
    print(f'Device {device_id}: native and video VAE attention checks passed', flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    report = dict(device_execution_passed=False, inference_passed=False, devices={})
    try:
        import torch
        devices = [int(item) for item in args.devices.split(',')]
        if not devices or min(devices) < 0 or max(devices) >= torch.cuda.device_count():
            raise RuntimeError(f'Requested {devices}; this session exposes {torch.cuda.device_count()} devices')
        from songshan_h3.attention_bridge import install
        install()
        for device_id in devices:
            report['devices'][str(device_id)] = check_device(device_id)
            print(f'Device {device_id}: dense, grouped, strided and varlen attention passed', flush=True)
        report['device_execution_passed'] = True
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        return 1
    finally:
        path = ROOT / 'artifacts/attention-bridge-device.json'
        path.write_text(json.dumps(report, indent=2) + '\n')
        print('Report: ' + str(path))


if __name__ == '__main__':
    raise SystemExit(main())
