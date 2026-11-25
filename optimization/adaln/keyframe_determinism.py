"""在关键帧编码期间启用 PyTorch 的确定性 DNN 实现。"""
from functools import wraps
from threading import RLock

import torch


_dnn_flags = RLock()


def wrap_keyframe_encode(encode):
    @wraps(encode)
    def deterministic(vae, pixels, *args, **kwargs):
        if not pixels.is_cuda or pixels.ndim != 5 or pixels.shape[2] != 1:
            return encode(vae, pixels, *args, **kwargs)
        with _dnn_flags:
            previous = torch.backends.cudnn.deterministic
            torch.backends.cudnn.deterministic = True
            try:
                return encode(vae, pixels, *args, **kwargs)
            finally:
                torch.backends.cudnn.deterministic = previous

    return deterministic
