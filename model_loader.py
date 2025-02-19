import torch
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from constants import STRENGTH, GUIDANCE_SCALE, NUM_INFERENCE_STEPS


def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "benjamin-paine/stable-diffusion-v1-5-inpainting",
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        generator = torch.Generator(device=device).manual_seed(0)
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(
            "benjamin-paine/stable-diffusion-v1-5-inpainting",
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        generator = torch.Generator(device=device).manual_seed(0)

    return pipe, generator, device
