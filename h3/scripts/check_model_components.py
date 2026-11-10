#!/usr/bin/env python3
"""Load and exercise H3 preprocessing components without model weights or GPUs."""
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from songshan_h3.preprocessing import check_model_components
    from prepare_public_sample import validate_local
    report = dict(passed=False, model_weights_loaded=False, inference_passed=False)
    try:
        manifest = json.loads((ROOT / 'artifacts/model-download.json').read_text())
        case = validate_local()
        report.update(check_model_components(manifest['model_path'], case['first_frame']['path']))
        print(json.dumps(report, indent=2))
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        return 1
    finally:
        (ROOT / 'artifacts/model-components-check.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
