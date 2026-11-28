"""保持原时空分块和拼接顺序，将独立空间 tile 分派到各卡。"""

from concurrent.futures import ThreadPoolExecutor

import torch

from video_decode_parallel import decoder_replicas


def schedule_spatial(vae, latents, num_chunks, devices, clip_batch):
    if clip_batch != 1 or not vae.use_tiling:
        raise ValueError("空间调度诊断需要原 VAE tiling 和单窗口 batch")
    primary = latents.device
    ratio = vae.spatial_compression_ratio
    y_starts, y_lengths, y_overlaps = vae._split_tiles(
        latents.shape[-2] * ratio, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_starts, x_lengths, x_overlaps = vae._split_tiles(
        latents.shape[-1] * ratio, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    jobs = [
        (clip, y_rank, x_rank, slice(y // ratio, (y + height) // ratio),
         slice(x // ratio, (x + width) // ratio))
        for clip in range(num_chunks)
        for y_rank, (y, height) in enumerate(zip(y_starts, y_lengths, strict=True))
        for x_rank, (x, width) in enumerate(zip(x_starts, x_lengths, strict=True))
    ]
    devices = tuple(devices[:len(jobs)])
    if not devices or devices[0] != primary:
        raise ValueError("第一张 VAE 卡必须持有输入和拼接结果")
    torch.cuda.current_stream(primary).synchronize()
    replicas = decoder_replicas(vae, devices)
    for device in devices:
        torch.cuda.current_stream(device).synchronize()

    @torch.inference_mode()
    def worker(rank, replica, device):
        completed = []
        with torch.cuda.device(device):
            local = latents.to(device)
            for clip, y_rank, x_rank, y_slice, x_slice in jobs[rank::len(devices)]:
                start = clip * vae.tokens_chunk_size
                stop = start + vae.tokens_chunk_size + vae.token_overlap
                tile = local[:, :, start:stop, y_slice, x_slice]
                decoded = replica.decoder(replica.post_quant_conv(tile))
                completed.append((clip, y_rank, x_rank, decoded))
            torch.cuda.current_stream(device).synchronize()
        return completed

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        tasks = [
            pool.submit(worker, rank, replica, device)
            for rank, (replica, device) in enumerate(zip(replicas, devices, strict=True))
        ]
        raw = [task.result() for task in tasks]
    tiles = [[[None for _ in x_starts] for _ in y_starts] for _ in range(num_chunks)]
    for completed in raw:
        for clip, y_rank, x_rank, decoded in completed:
            tiles[clip][y_rank][x_rank] = decoded.to(primary)
    return [vae._stitch_tiles(clip, y_overlaps, x_overlaps) for clip in tiles]
