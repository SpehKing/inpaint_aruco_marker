import cv2
import numpy as np
from PIL import Image


def inpaint_region(
    pipe,
    generator,
    cropped_img,
    cropped_mask,
    guidance_scale,
    num_inference_steps,
    strength,
):
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cropped_img_pil = Image.fromarray(cropped_img_rgb)
    cropped_mask_pil = Image.fromarray(cropped_mask).convert("L")

    inpainted_result = pipe(
        prompt="",
        image=cropped_img_pil,
        mask_image=cropped_mask_pil,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
    ).images[0]

    inpainted_np = cv2.cvtColor(np.array(inpainted_result), cv2.COLOR_RGB2BGR)
    return inpainted_np
