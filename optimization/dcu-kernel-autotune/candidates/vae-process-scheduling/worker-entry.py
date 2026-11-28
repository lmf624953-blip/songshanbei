#!/usr/bin/env python3
"""独立启动 VAE worker，避免 spawn 重新执行无 main guard 的正式启动器。"""

import argparse
import os
from multiprocessing.connection import Client

import torch.multiprocessing as mp

from bench import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--auth-fd", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    with os.fdopen(args.auth_fd, "rb") as stream:
        authentication = stream.read()
    assert len(authentication) == 32
    mp.current_process().authkey = authentication
    connection = Client(args.socket, family="AF_UNIX", authkey=authentication)
    connection.send(("connected", args.rank))
    worker(args.model, args.rank, connection, args.log)


if __name__ == "__main__":
    main()
