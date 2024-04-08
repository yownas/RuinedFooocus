import torch
import os
import einops
from latent_preview import TAESD, TAESDPreviewerImpl
import numpy as np


def clean_prompt_cond_caches():
    conditions = {}
    conditions["+"] = {}
    conditions["-"] = {}
    conditions["switch"] = {}
    conditions["+"]["text"] = None
    conditions["+"]["cache"] = None
    conditions["-"]["text"] = None
    conditions["-"]["cache"] = None
    conditions["switch"]["text"] = None
    conditions["switch"]["cache"] = None
    return conditions


def load_all_keywords(loras):
    lora_prompt_addition = ""
    return lora_prompt_addition


def set_timestep_range(conditioning, start, end):
    c = []
    for t in conditioning:
        if "pooled_output" in t:
            t["start_percent"] = start
            t["end_percent"] = end

    return conditioning


def get_previewer(device, latent_format):
    taesd_decoder_path = os.path.abspath(
        os.path.realpath(
            os.path.join("models", "vae_approx", latent_format.taesd_decoder_name)
        )
    )

    if not os.path.exists(taesd_decoder_path):
        print(
            f"Warning: TAESD previews enabled, but could not find {taesd_decoder_path}"
        )
        return None

    taesd = TAESD(None, taesd_decoder_path).to(device)

    def preview_function(x0, step, total_steps):
        global cv2_is_top
        with torch.torch.inference_mode():
            x_sample = (
                taesd.taesd_decoder(
                    torch.nn.functional.avg_pool2d(x0, kernel_size=(2, 2))
                ).detach()
                * 255.0
            )
            x_sample = einops.rearrange(x_sample, "b c h w -> b h w c")
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample[0]

    taesd.preview = preview_function

    return taesd
