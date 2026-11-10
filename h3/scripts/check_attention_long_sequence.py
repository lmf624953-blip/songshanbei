#!/usr/bin/env python3
"""Check production-size attention against a small query-row math oracle."""
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from check_attention_bridge import math_attention_reference
    from songshan_h3.attention_bridge import _launch
    report = dict(diagnostic_only=True, passed=False, inference_passed=False, cases=[])
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('This session does not expose an accelerator')
        torch.manual_seed(42)
        for length, heads, dim in ((28228, 32, 64), (32769, 7, 128)):
            print(f'Checking long attention: N={length}, heads={heads}, dimension={dim}', flush=True)
            q, k, v = (torch.randn(1, length, heads, dim, device='cuda:0', dtype=torch.bfloat16)
                        for _ in range(3))
            out = torch.empty_like(q)
            _launch(out, q, k, v, dim**-0.5, False)
            indices = torch.tensor([0, 1, 63, 64, length // 2, length - 1], device=q.device)
            expected = math_attention_reference(q.index_select(1, indices).permute(0, 2, 1, 3),
                                                k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
            actual = out.index_select(1, indices).permute(0, 2, 1, 3)
            torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.03)
            report['cases'].append(dict(sequence_length=length, heads=heads, head_dim=dim,
                                        checked_rows=indices.tolist(),
                                        max_abs_error=float((actual.float() - expected.float()).abs().max())))
            del q, k, v, out, expected, actual
        report['passed'] = True
        print('Long-sequence attention sampled-row comparisons: PASS', flush=True)
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        return 1
    finally:
        (ROOT / 'artifacts/attention-long-sequence.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
