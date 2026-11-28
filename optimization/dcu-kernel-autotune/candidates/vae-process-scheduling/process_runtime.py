"""独立 subprocess 的 VAE 调度，可在原入口与常驻配对中按请求切换。"""

import atexit
import functools
import importlib.util
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import RLock
from time import monotonic
from types import FunctionType
import weakref

import torch
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("songshan_vae_process_bench", HERE / "bench.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


class Process:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self.process.terminate()


class Workers(core.Workers):
    def __init__(self, model, count, directory):
        self.connections = [None] * count
        self.processes = []
        self.request, self.last_samples = 0, []
        self.socket_directory = tempfile.TemporaryDirectory(prefix="h3-vae-")
        authentication = bytes(mp.current_process().authkey)
        assert len(authentication) == 32
        directory.mkdir(parents=True, exist_ok=True)
        address = str(Path(self.socket_directory.name) / "pool")
        listener = Listener(address, family="AF_UNIX", authkey=authentication)
        listener._listener._socket.settimeout(240)
        try:
            for rank in range(count):
                read_fd, write_fd = os.pipe()
                try:
                    command = [sys.executable, "-u", str(HERE / "worker-entry.py"),
                               "--model", str(model), "--rank", str(rank), "--socket", address,
                               "--auth-fd", str(read_fd), "--log", str(directory / f"worker-{rank}.log")]
                    process = subprocess.Popen(command, pass_fds=(read_fd,))
                    self.processes.append(Process(process))
                    os.write(write_fd, authentication)
                finally:
                    os.close(read_fd)
                    os.close(write_fd)
            for _ in range(count):
                connection = listener.accept()
                assert connection.poll(60)
                kind, rank = connection.recv()
                assert kind == "connected" and 0 <= rank < count and self.connections[rank] is None
                self.connections[rank] = connection
            (directory / "worker-pids.json").write_text(json.dumps([p.pid for p in self.processes]) + "\n")
            for rank in range(count):
                result = self.receive(rank)
                assert result == ("ready", rank, "torch.bfloat16"), result
                print("[vae_process] ready", rank, flush=True)
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()

    def close(self):
        # 初始化失败时部分连接可能尚未建立。
        for process, connection in zip(self.processes, self.connections):
            if process.is_alive() and connection is not None:
                try:
                    connection.send(("stop",))
                except (BrokenPipeError, EOFError):
                    pass
        deadline = monotonic() + 20
        for process in self.processes:
            process.join(max(0, deadline - monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(3)
        for connection in self.connections:
            if connection is not None:
                connection.close()
        self.socket_directory.cleanup()


def install(control, count, directory):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from video_decode_parallel import decode_parallel

    if not 2 <= count <= torch.cuda.device_count():
        raise ValueError("VAE进程数必须在2和可见设备数之间")
    state = dict(enabled=True, variant="candidate", base=AutoencoderKLMiniMaxH3._decode,
                 pool=None, owner=None)
    lock = RLock()
    devices = tuple(torch.device("cuda", rank) for rank in range(count))

    def schedule(vae, *arguments):
        if state["pool"] is None:
            source = Path(vae.config._name_or_path).resolve(strict=True)
            if (source / "vae/config.json").is_file():
                model = source
            elif source.name == "vae" and (source / "config.json").is_file():
                model = source.parent
            else:
                raise ValueError(f"无法定位当前VAE的原权重目录：{source}")
            state["pool"] = Workers(model, count, Path(directory))
            state["owner"] = weakref.ref(vae)
        if state["owner"]() is not vae:
            raise RuntimeError("VAE实例发生变化，需建立新的worker池")
        return state["pool"].schedule(vae, *arguments)

    # 只为原Python函数绑定不同的宿主调度器，原拼接字节码与数学计算不变。
    namespace = dict(decode_parallel.__globals__, schedule_clips=schedule)
    decode = FunctionType(decode_parallel.__code__, namespace, decode_parallel.__name__,
                          decode_parallel.__defaults__, decode_parallel.__closure__)
    decode.__kwdefaults__ = dict(decode_parallel.__kwdefaults__ or {})

    @functools.wraps(AutoencoderKLMiniMaxH3._decode)
    def dispatch(vae, z):
        if vae.training or not z.is_cuda or z.device != devices[0]:
            return state["base"](vae, z)
        with lock:
            return decode(vae, z, devices, serial=state["base"], clip_batch=1)

    def enabled(value):
        state["enabled"] = bool(value)
        AutoencoderKLMiniMaxH3._decode = dispatch if value and state["variant"] != "original" else state["base"]

    original_select = control.select_variant

    def select(variant):
        original_select(variant)
        state["variant"] = variant
        state["base"] = AutoencoderKLMiniMaxH3._decode
        enabled(state["enabled"])

    def stats():
        pool = state["pool"]
        return dict(enabled=state["enabled"], workers=count, ready=pool is not None,
                    requests=pool.request if pool else 0, last_samples=pool.last_samples if pool else [],
                    pids=[p.pid for p in pool.processes] if pool else [])

    def close():
        with lock:
            if state["pool"] is not None:
                state["pool"].close()
                print("[vae_process] worker exits", [p.exitcode for p in state["pool"].processes], flush=True)
                state["pool"] = None

    control.select_variant = select
    control.set_vae_process_enabled = enabled
    control.vae_process_stats = stats
    atexit.register(close)
    enabled(True)
    return dict(close=close, stats=stats, enabled=enabled)
