#!/usr/bin/env python3
"""Sample total device VRAM, including native allocations outside PyTorch."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--until", required=True, type=Path)
    parser.add_argument("--interval", default=5, type=float)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    with args.output.open("x", buffering=1) as stream:
        while not args.until.exists():
            raw = subprocess.check_output(
                ["hy-smi", "--showmeminfo", "vram", "--json"], text=True, timeout=15,
            )
            used = {
                card: int(fields["vram Total Used Memory (MiB)"])
                for card, fields in json.loads(raw).items()
            }
            stream.write(json.dumps({"epoch_ns": time.time_ns(), "used_mib": used}) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
