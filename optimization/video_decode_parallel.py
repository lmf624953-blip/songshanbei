"""Schedule the pinned video VAE's independent temporal clips across BW1000s."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import os

import torch
from torch.cuda import comm


def replicate_decoders(vae, devices):
    # Preserve registered parameters and dtypes used by the decoder's forward methods.
    names = ("decoder", "post_quant_conv")
    modules = tuple(getattr(vae, name) for name in names)
    parameters, buffers = {}, {}
    for module in modules:
        parameters.update((id(p), p.detach()) for p in module.parameters())
        buffers.update((id(buffer), buffer) for buffer in module.buffers())
    sources = parameters | buffers
    gpu = {key: tensor for key, tensor in sources.items() if tensor.device == devices[0]}
    others = {key: tensor for key, tensor in sources.items() if key not in gpu}
    copies = comm.broadcast_coalesced(list(gpu.values()), devices, buffer_size=32 * 1024 * 1024)
    replicas = [vae]
    for device, moved in zip(devices[1:], copies[1:]):
        memo = dict(zip(gpu, moved))
        memo.update((key, tensor.to(device)) for key, tensor in others.items())
        for key in parameters:
            memo[key] = torch.nn.Parameter(memo[key], requires_grad=False)
        replica = copy(vae)
        replica._modules = vae._modules.copy()
        for name, module in zip(names, modules):
            setattr(replica, name, deepcopy(module, memo))
        replicas.append(replica)
    return replicas


def decoder_replicas(vae, devices):
    if os.environ.get("H3_VAE_CACHE_REPLICAS") != "1" or vae.training:
        return replicate_decoders(vae, devices)
    modules = tuple(getattr(vae, name) for name in ("decoder", "post_quant_conv"))
    tensors = tuple(tensor for module in modules for tensor in (*module.parameters(), *module.buffers()))
    try:
        versions = tuple((id(t), t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()),
                          t.dtype, t.device) for t in tensors)
    except RuntimeError:
        # 未记录版本的 inference tensor 无法核验后续原地改权重，沿用逐请求复制。
        return replicate_decoders(vae, devices)
    key = (tuple(devices), versions, tuple(child.training for module in modules for child in module.modules()))
    cached = getattr(vae, "_songshan_decoder_replica_cache", None)
    if cached is not None and cached[0] == key:
        return [vae, *cached[1]]
    replicas = replicate_decoders(vae, devices)
    # 仅保留 decoder 权重副本；不保存输入、latent、窗口或生成输出。
    # 不把 vae 自身存入属性，避免形成引用环。
    vae._songshan_decoder_replica_cache = (key, replicas[1:])
    return replicas


def schedule_clips(vae, z, num_chunks, devices, clip_batch):
    primary = z.device
    devices = devices[:num_chunks]
    torch.cuda.current_stream(primary).synchronize()
    replicas = decoder_replicas(vae, devices) if len(devices) > 1 else [vae]
    for device in devices:
        torch.cuda.current_stream(device).synchronize()

    @torch.inference_mode()
    def decode_jobs(decoder, device, worker):
        with torch.cuda.device(device):
            jobs = []
            indices = list(range(worker, num_chunks, len(devices)))
            for offset in range(0, len(indices), clip_batch):
                group = indices[offset:offset + clip_batch]
                windows = []
                for i in group:
                    start = i * vae.tokens_chunk_size
                    stop = start + vae.tokens_chunk_size + vae.token_overlap
                    windows.append(z[:, :, start:stop].to(device))
                if len(windows) == 1 or any(w.shape != windows[0].shape for w in windows[1:]):
                    decoded = [decoder._decode_clip(window) for window in windows]
                else:
                    decoded = decoder._decode_clip(torch.cat(windows, dim=0)).split(z.shape[0], dim=0)
                jobs.extend(zip(group, decoded, strict=True))
            torch.cuda.current_stream(device).synchronize()
            return jobs

    if len(devices) == 1:
        completed = [decode_jobs(vae, primary, 0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as workers:
            tasks = [workers.submit(decode_jobs, decoder, device, i)
                     for i, (decoder, device) in enumerate(zip(replicas, devices))]
            completed = [task.result() for task in tasks]
    clips = [None] * num_chunks
    for jobs in completed:
        for i, clip in jobs:
            clips[i] = clip.to(primary)
    return clips


def decode_parallel(vae, z, devices, serial=None, *, clip_batch=1):
    # The window sizes, padding, clip order and blend follow the pinned _decode.
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio
    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if num_chunks < 2 or (len(devices) < 2 and clip_batch == 1):
        return vae._decode(z) if serial is None else serial(vae, z)
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded_chunks = []
    overlap = None
    for clip in schedule_clips(vae, z, num_chunks, devices, clip_batch):
        for j in range(int(token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip[:, :, frame_start:frame_start + chunk_num_frames]
            chunk = chunk[:, :, vae.frame_pre_padding:]
            if j == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk
    if overlap is not None:
        decoded_chunks.append(overlap)
    decoded = torch.cat(decoded_chunks, dim=2)
    if pad_tokens > 0:
        intra_tail = vae.config.clip_length % temporal_ratio
        before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        decoded = decoded[:, :, :-pad_frames]
    return decoded


def install_parallel_decode(device_ids, clip_batch=1):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    serial = AutoencoderKLMiniMaxH3._decode
    devices = tuple(torch.device("cuda", i) for i in device_ids)
    if (not devices or len(set(devices)) != len(devices) or clip_batch < 1
        or (len(devices) == 1 and clip_batch == 1)):
        raise ValueError("Video decode scheduling needs distinct GPUs or a clip batch greater than one")

    def dispatch(vae, z):
        if not z.is_cuda or z.device != devices[0]:
            return serial(vae, z)
        return decode_parallel(vae, z, devices, serial, clip_batch=clip_batch)

    AutoencoderKLMiniMaxH3._decode = dispatch
