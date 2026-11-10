"""Use the existing NineToothed operator for the native H3 video VAE."""
import torch

from .attention_bridge import _launch


def video_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False,
                    return_lse=False, backend=None, parallel_config=None):
    if attn_mask is not None or dropout_p or is_causal or return_lse or parallel_config is not None:
        raise ValueError('H3 video VAE adapter supports unmasked, noncausal inference attention')
    if not enable_gqa and query.shape[2] != key.shape[2]:
        raise ValueError('Unequal head counts require enable_gqa=True')
    output = torch.empty_like(query)
    _launch(output, query, key, value,
            query.shape[-1]**-0.5 if scale is None else float(scale), False)
    return output


def install_video_attention():
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as video_vae
    # This module has one attention dispatch call, in MiniMaxH3VideoAttnProcessor.
    # Other Diffusers models and the legacy reference process retain their backend.
    video_vae.dispatch_attention_fn = video_attention
