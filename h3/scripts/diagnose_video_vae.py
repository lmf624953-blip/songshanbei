#!/usr/bin/env python3
"""Isolated VAE round trip, not a scored FL2VA generation or benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('fp32', 'bf16', 'reference-mixed', 'rounded-fp32'), default='fp32')
    parser.add_argument('--attention', choices=('math', 'ninetoothed'), default='math')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--reference', type=Path, help='Compare reconstruction with another diagnostic directory')
    parser.add_argument('--decode-path', choices=('raw', 'public'), default='raw')
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--tiling', action='store_true', help='Use the released spatial tiling, as in production')
    args = parser.parse_args()
    output = ROOT / 'artifacts/vae-diagnostics' / (args.device.replace(':', '-') + '_' + args.precision + '_' + args.attention)
    if args.decode_path != 'raw' or (args.height, args.width) != (128, 224):
        output = output.with_name(output.name + f'_{args.decode_path}_{args.height}x{args.width}')
    if args.tiling:
        output = output.with_name(output.name + '_tiled')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  diagnostic_only=True, full_inference_passed=False, passed=False)
    try:
        if args.reference:
            baseline = json.loads((args.reference / 'report.json').read_text())['arguments']
            for key, default in (('decode_path', 'raw'), ('height', 128), ('width', 224)):
                if baseline.get(key, default) != getattr(args, key):
                    raise ValueError(f'Incompatible reference {key}: {baseline.get(key, default)} vs {getattr(args, key)}')
        import numpy as np
        import torch
        from PIL import Image
        from diffusers import AutoencoderKLMiniMaxH3
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from skimage.metrics import structural_similarity

        torch.set_num_threads(args.threads)
        manifest = json.loads((ROOT / 'artifacts/model-download.json').read_text())
        from prepare_public_sample import validate_local
        case = validate_local()
        dtype = torch.float32 if args.precision in ('fp32', 'rounded-fp32') else torch.bfloat16
        started = time.monotonic()
        print('Loading video VAE only, no conditioner or H3 transformer weights', flush=True)
        vae, loading = AutoencoderKLMiniMaxH3.from_pretrained(
            str(Path(manifest['model_path']) / 'vae'), torch_dtype=dtype,
            local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
        report['loading_info'] = loading
        vae.to(device=args.device)
        if args.precision == 'bf16':
            vae.to(dtype=dtype)  # Reproduce the generation entry's blanket cast.
        elif args.precision == 'rounded-fp32':
            # Isolate weight rounding from CPU BF16 arithmetic performance.
            vae.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        vae.eval().requires_grad_(False)
        if not args.tiling:
            vae.disable_tiling()
        report['load_seconds'] = time.monotonic() - started
        report['model_revision'] = manifest['revision']
        report['parameter_dtypes'] = {name: str(next(module.parameters()).dtype)
                                      for name, module in (('encoder', vae.encoder),
                                                           ('decoder', vae.decoder),
                                                           ('post_quant_conv', vae.post_quant_conv))}
        report['precision_policy'] = {name: str(parameter.dtype) for name, parameter in vae.named_parameters()
                                      if name in ('decoder.transformer_blocks.0.norm1.weight',
                                                  'decoder.transformer_blocks.0.scale1',
                                                  'decoder.proj_out.weight')}
        if min(args.height, args.width) < 32 or args.height % 16 or args.width % 16:
            raise ValueError('Diagnostic image dimensions must be positive multiples of 16, at least 32')
        image = Image.open(case['first_frame']['path']).convert('RGB').resize((args.width, args.height), Image.Resampling.LANCZOS)
        image.save(output / 'input.png')
        original = np.asarray(image)
        pixels = torch.from_numpy(original.copy()).permute(2, 0, 1)[None, :, None].float().to(args.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1, 1)
        pixels = (pixels / 255.0 - mean) / std
        if args.attention == 'ninetoothed':
            if args.device == 'cpu':
                raise ValueError('NineToothed device diagnostic requires a Hygon device')
            from songshan_h3.diffusers_attention import install_video_attention
            install_video_attention()
        print('Encoding one diagnostic frame; decoder path=' + args.decode_path, flush=True)
        started = time.monotonic()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            latents = vae.encode(pixels).latent_dist.mode()
            if args.decode_path == 'public':
                # Repeated real-image latents isolate the full public decoder,
                # including temporal chunk assembly, from the DiT predictions.
                decoder_latents = latents.repeat(1, 1, vae.tokens_chunk_size, 1, 1)
                reconstruction = vae.decode(decoder_latents, return_dict=False)[0]
            else:
                # Bypass temporal clip stitching to isolate the spatial path.
                reconstruction = vae._decode_clip(latents.to(next(vae.post_quant_conv.parameters()).dtype))
        if args.device != 'cpu':
            torch.cuda.synchronize()
        report['compute_seconds'] = time.monotonic() - started
        report['latent_shape'] = list(latents.shape)
        report['decoded_shape'] = list(reconstruction.shape)
        report['latents_finite'] = bool(torch.isfinite(latents).all())
        report['decoded_finite'] = bool(torch.isfinite(reconstruction).all())
        from safetensors.torch import save_file
        save_file({'latents': latents.float().cpu().contiguous(),
                   'decoded': reconstruction.float().cpu().contiguous()}, str(output / 'tensors.safetensors'))
        # Match the released pipeline's ImageNet convention and the decoder's
        # three leading padding frames, rather than comparing a padding frame.
        frame_index = 0 if args.decode_path == 'public' else vae.frame_pre_padding
        report['decoded_frame_index'] = frame_index
        rgb = (reconstruction.float() * std + mean).clamp(0, 1)
        decoded = (rgb[0, :, frame_index].cpu().permute(1, 2, 0) * 255).round().byte().numpy()
        Image.fromarray(decoded).save(output / 'reconstructed.png')
        if args.decode_path == 'public':
            for index in sorted({0, rgb.shape[2] // 2, rgb.shape[2] - 1}):
                frame = (rgb[0, :, index].cpu().permute(1, 2, 0) * 255).round().byte().numpy()
                Image.fromarray(frame).save(output / f'decoded-frame-{index:03}.png')
        report['first_frame_ssim'] = float(structural_similarity(original, decoded, channel_axis=-1, data_range=255))
        report['passed'] = report['latents_finite'] and report['decoded_finite']
        report['quality_threshold_applied'] = False
        report['note'] = 'Round-trip execution only; low-resolution SSIM is diagnostic, not a competition score.'
        if args.reference:
            reference = np.asarray(Image.open(args.reference / 'reconstructed.png').convert('RGB'))
            report['reference_ssim'] = float(structural_similarity(reference, decoded, channel_axis=-1, data_range=255))
            report['reference_consistency_passed'] = report['reference_ssim'] >= 0.9
            report['passed'] = report['passed'] and report['reference_consistency_passed']
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report['passed'] else 1
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        return 1
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        print('Diagnostic output: ' + str(output), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
