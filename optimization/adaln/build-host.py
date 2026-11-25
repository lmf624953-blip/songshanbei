#!/usr/bin/env python3
"""Build the AdaLN host shim with the installed attention bridge ABI."""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[1] / "runtime").resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    output = args.destination.resolve() / "adaln-host.so"
    template = json.loads((RUNTIME / "artifacts/attention-bridge-build.json").read_text())["command"]
    command = list(template)
    command[command.index(str(RUNTIME / "songshan_h3/attention_bridge.cc"))] = str(HERE / "adaln_host.cc")
    command[command.index("-o") + 1] = str(output)
    subprocess.run(command, check=True)
    (output.parent / "host-build-command.json").write_text(json.dumps(command, indent=2) + "\n")


if __name__ == "__main__":
    main()
