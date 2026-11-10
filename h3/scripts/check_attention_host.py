#!/usr/bin/env python3
"""CPU checks for the small math oracle and the video VAE dispatch wiring."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_attention_bridge import math_attention_reference


def backend_flags():
    cuda = torch.backends.cuda
    return dict(flash=cuda.flash_sdp_enabled(), math=cuda.math_sdp_enabled(),
                efficient=cuda.mem_efficient_sdp_enabled(), cudnn=cuda.cudnn_sdp_enabled())


def main():
    from songshan_h3.diffusers_attention import install_video_attention
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as video
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3_audio as audio
    from torch.nn.attention import sdpa_kernel, SDPBackend

    report = dict(math_reference_passed=False, video_dispatch_wiring_passed=False,
                  device_execution_passed=False, inference_passed=False)
    try:
        previous = backend_flags()
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def guarded_sdpa(*args, **kwargs):
            if backend_flags() != dict(flash=False, math=True, efficient=False, cudnn=False):
                raise AssertionError('Reference SDPA may select an unavailable external library')
            return original_sdpa(*args, **kwargs)

        cases = []
        for batch, nq, nk, hq, hk, dim, causal in (
                (1, 4, 4, 2, 2, 128, False), (1, 7, 71, 4, 2, 72, True),
                (1, 5, 5, 2, 2, 128, True), (2, 67, 67, 4, 4, 64, False)):
            q = torch.randn(batch, hq, nq, dim, dtype=torch.bfloat16)
            k, v = (torch.randn(batch, hk, nk, dim, dtype=torch.bfloat16) for _ in range(2))
            mask = (torch.arange(nq)[:, None] + nk - nq >= torch.arange(nk)[None, :]) if causal else None
            with patch.object(torch.nn.functional, 'scaled_dot_product_attention', guarded_sdpa):
                result = math_attention_reference(q, k, v, attn_mask=mask, enable_gqa=(hq != hk))
            kk, vv = (t.double().repeat_interleave(hq // hk, dim=1) for t in (k, v))
            scores = q.double() @ kk.transpose(-2, -1) * dim**-0.5
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            expected = (scores.softmax(-1) @ vv).to(q.dtype)
            torch.testing.assert_close(result, expected, rtol=0.04, atol=0.03)
            assert backend_flags() == previous, 'Reference backend flags were not restored'
            cases.append(dict(batch=batch, head_dim=dim, causal=causal, grouped_heads=(hq != hk)))
        report.update(math_reference_passed=True, reference_cases=cases)

        # Exercise the unchanged processor and projections. Only the device
        # launcher is substituted by a CPU math oracle in this wiring check.
        model = video.MiniMaxH3VideoAttention(dim=128, heads=2, dim_head=64).to(torch.bfloat16)
        inputs = torch.randn(2, 7, 128, dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.MATH):
            expected = model(inputs)
        original_video_dispatch = video.dispatch_attention_fn
        original_audio_dispatch = audio.dispatch_attention_fn

        def host_launch(out, q, k, v, scale, causal):
            assert scale == 64**-0.5 and not causal
            out.copy_(math_attention_reference(
                *(t.permute(0, 2, 1, 3) for t in (q, k, v))).permute(0, 2, 1, 3))

        try:
            install_video_attention()
            with patch('songshan_h3.diffusers_attention._launch', host_launch):
                actual = model(inputs)
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
            assert audio.dispatch_attention_fn is original_audio_dispatch
        finally:
            video.dispatch_attention_fn = original_video_dispatch
        report['video_dispatch_wiring_passed'] = True
        print('Math reference selection, independent numerical oracle and video VAE wiring: PASS (CPU only)')
        return 0
    except Exception:
        import traceback
        report['error'] = traceback.format_exc()
        print(report['error'])
        return 1
    finally:
        (ROOT / 'artifacts/attention-host-check.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
