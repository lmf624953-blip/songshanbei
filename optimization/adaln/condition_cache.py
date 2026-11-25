"""调度条件编码器的设备和 KV 工作空间。"""
import torch

from infinilm.cache import StaticKVCacheConfig
from infinilm.multimodal import minimax_h3


def install(initial_capacity=None, tp_count=None):
    if initial_capacity is not None and initial_capacity < 1:
        raise ValueError("Initial encoder cache capacity must be positive")
    if tp_count not in (None, 2, 4, 8):
        raise ValueError("Use a supported conditioning encoder TP count")
    base = minimax_h3.MiniMaxH3ConditionEncoder

    class MiniMaxH3ConditionEncoder(base):
        def __init__(self, *args, max_cache_len=32768, **kwargs):
            self._condition_cache_limit = max_cache_len
            self._condition_cache_capacity = min(initial_capacity or max_cache_len, max_cache_len)
            self._condition_pipeline_device = None
            requested = kwargs.get("tp_device_ids")
            if tp_count and requested and len(requested) > tp_count and kwargs.get("tp_size", 1) == 1:
                kwargs["tp_device_ids"] = requested[-tp_count:]
                self._condition_pipeline_device = torch.device("cuda", requested[0])
                print("[condition_devices]", kwargs["tp_device_ids"], flush=True)
            super().__init__(*args, max_cache_len=self._condition_cache_capacity, **kwargs)

        @property
        def device(self):
            return self._condition_pipeline_device or super().device

        def encode(self, model_inputs):
            tokens = model_inputs["input_ids"]
            if tokens.ndim == 2 and tokens.shape[0] == 1:
                length = tokens.shape[1]
                if length > self._condition_cache_limit:
                    raise ValueError(
                        f"Condition sequence length {length} exceeds the model cache limit "
                        f"{self._condition_cache_limit}"
                    )
                if length > self._condition_cache_capacity:
                    capacity = min(1 << (length - 1).bit_length(), self._condition_cache_limit)
                    self.engine.reset_cache(StaticKVCacheConfig(1, capacity))
                    self._condition_cache_capacity = capacity
                    print(f"[condition_cache] capacity {capacity} for {length} tokens", flush=True)
            return super().encode(model_inputs)

    MiniMaxH3ConditionEncoder.__module__ = base.__module__
    minimax_h3.MiniMaxH3ConditionEncoder = MiniMaxH3ConditionEncoder
