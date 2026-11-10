"""CPU-only checks of the actual H3 tokenizer and multimodal processor."""
from importlib.metadata import version
from pathlib import Path


def check_vision_api():
    import torch
    import torchvision
    from torchvision.transforms.v2 import functional as tvf

    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
    scores = torch.tensor([0.9, 0.8])
    assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
    image = torch.zeros((3, 32, 48), dtype=torch.uint8)
    assert tuple(tvf.resize(image, [16, 24], antialias=True).shape) == (3, 16, 24)
    return dict(torch=version('torch'), torchvision=version('torchvision'),
                native_cpu_ops_passed=True, transforms_v2_passed=True)


def check_model_components(model_path, image_path):
    import torch
    from PIL import Image
    from diffusers import ModularPipeline
    from transformers import Qwen3VLProcessor

    vision_api = check_vision_api()
    model_path = Path(model_path)
    pipe = ModularPipeline.from_pretrained(str(model_path), local_files_only=True)
    pipe.load_components(names=['tokenizer', 'processor'],
                         pretrained_model_name_or_path=str(model_path), local_files_only=True)
    if pipe.processor is None:
        # Surface the original loading exception instead of a downstream None error.
        Qwen3VLProcessor.from_pretrained(str(model_path / 'processor'), local_files_only=True)
        raise RuntimeError('The H3 pipeline did not register its processor')
    if pipe.tokenizer is None:
        raise RuntimeError('The H3 pipeline did not load its tokenizer')
    processor = pipe.processor
    if processor.image_processor is None:
        raise RuntimeError('Qwen3VLProcessor.image_processor is None')
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        result = processor.image_processor(images=[image], return_tensors='pt')
    pixels, grid = result['pixel_values'], result['image_grid_thw']
    assert pixels.device.type == 'cpu' and pixels.ndim == 2
    assert tuple(grid.shape) == (1, 3) and (grid > 0).all()
    assert pixels.shape[0] == int(grid.prod(-1).sum())
    assert torch.isfinite(pixels).all()
    ids = pipe.tokenizer('H3 preprocessing check', add_special_tokens=False)['input_ids']
    tags = processor.create_mm_token_type_ids([ids])
    assert len(tags) == 1 and len(tags[0]) == len(ids)
    return dict(passed=True, model_weights_loaded=False, device_execution_passed=False,
                vision_api=vision_api, processor_type=type(processor).__name__,
                image_processor_type=type(processor.image_processor).__name__,
                tokenizer_type=type(pipe.tokenizer).__name__,
                pixel_values_shape=list(pixels.shape), image_grid_thw=grid.tolist(),
                pixel_values_dtype=str(pixels.dtype))
