import torch
import os
import einops
from latent_preview import Latent2RGBPreviewer
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


def set_timestep_range(conditioning, start, end):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]

        if "pooled_output" in n[1]:
            n[1]["start_percent"] = start
            n[1]["end_percent"] = end

        c.append(n)

    return c


def get_previewer(device, latent_format):
    previewer = Latent2RGBPreviewer(
        latent_rgb_factors=latent_format.latent_rgb_factors,
        latent_rgb_factors_bias=latent_format.latent_rgb_factors_bias,
        latent_rgb_factors_reshape=latent_format.latent_rgb_factors_reshape
    )
    def preview_function(x0, step, total_steps):
        return previewer.decode_latent_to_preview(x0)
    previewer.preview = preview_function
    return previewer
