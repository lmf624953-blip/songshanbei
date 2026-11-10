#!/usr/bin/env python3
"""Inspect an existing run without regenerating it or rewriting its run record."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songshan_h3.output_validation import validate_generated_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    run = json.loads((directory / 'run.json').read_text())
    report = validate_generated_media(directory / 'video.mp4', run['configuration'])
    report['process_exit_code'] = run.get('returncode')
    report['clean_exit'] = run.get('returncode') == 0
    report['generation_completed_marker'] = ('[done] Saved to: ' + str(directory / 'video.mp4')
                                              in (directory / 'generation.log').read_text(errors='replace'))
    report['source_run_record'] = str(directory / 'run.json')
    (directory / 'video-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['video_specs_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
