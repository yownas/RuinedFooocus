import gc
import numpy as np
import os
import torch
import warnings

import modules.core as core
import modules.path

from PIL import Image, ImageOps

from modules.settings import default_settings
from modules.util import suppress_stdout

import warnings
import time

from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from diffusers.utils import numpy_to_pil
from torch import nn

#warnings.filterwarnings("ignore", category=UserWarning)


def clean_prompt_cond_caches():
    return

wuerst_prior_pipeline = None
wuerst_decoder_pipeline = None

def load_base_model(model):
    global wuerst_prior_pipeline, wuerst_decoder_pipeline
    device = "cuda"
    dtype = torch.float16
    if wuerst_prior_pipeline is None:
        # https://huggingface.co/warp-ai/wuerstchen-prior-model-interpolated/tree/main
        wuerst_prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
            "warp-ai/wuerstchen-prior", torch_dtype=dtype).to(device)
        wuerst_prior_pipeline.enable_attention_slicing()
        wuerst_prior_pipeline.enable_model_cpu_offload()
        wuerst_prior_pipeline.enable_xformers_memory_efficient_attention()
        #wuerst_prior_pipeline.prior = torch.compile(wuerst_prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
    if wuerst_decoder_pipeline is None:
        wuerst_decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
            "warp-ai/wuerstchen", torch_dtype=dtype).to(device)
        wuerst_decoder_pipeline.enable_attention_slicing()
        wuerst_decoder_pipeline.enable_model_cpu_offload()
        wuerst_decoder_pipeline.enable_xformers_memory_efficient_attention()
        #wuerst_decoder_pipeline.decoder = torch.compile(wuerst_decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)

def load_refiner_model(model):
    return

def load_loras(loras):
    return

# Effnet 16x16 to 64x64 previewer
# From https://github.com/camenduru/Wuerstchen-hf
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1), # 36 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.ConvTranspose2d(c_hidden, c_hidden//2, kernel_size=2, stride=2), # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden//2),

            nn.Conv2d(c_hidden//2, c_hidden//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden//2),

            nn.ConvTranspose2d(c_hidden//2, c_hidden//4, kernel_size=2, stride=2), # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden//4),

            nn.Conv2d(c_hidden//4, c_hidden//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden//4),

            nn.Conv2d(c_hidden//4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)

@torch.no_grad()
def process(
    positive_prompt,
    negative_prompt,
    steps,
    switch,
    width,
    height,
    image_seed,
    start_step,
    denoise,
    cfg,
    base_clip_skip,
    refiner_clip_skip,
    sampler_name,
    scheduler,
    callback,
):
    global wuerst_prior_pipeline, wuerst_decoder_pipeline

    device = torch.device("cuda:0")
    dtype = torch.float16

    if wuerst_prior_pipeline is None or wuerst_decoder_pipeline is None:
        load_base_model(None)

    seed_gen = torch.Generator().manual_seed(image_seed)

    previewer = Previewer()
    # https://huggingface.co/spaces/warp-ai/Wuerstchen/resolve/main/previewer/text2img_wurstchen_b_v1_previewer_100k.pt
    previewer.load_state_dict(torch.load("models/wuerstchen/text2img_wurstchen_b_v1_previewer_100k.pt")["state_dict"])
    previewer.eval().requires_grad_(False).to(device).to(dtype)

    def callback_prior(i, t, latents):
        if callback is not None:
            output = previewer(latents)
            output = numpy_to_pil(output.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
            callback(i, 0, 0, steps, output[0])
        #return output

    prior = wuerst_prior_pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        height=round(height/128)*128,
        width=round(width/128)*128,
        timesteps=DEFAULT_STAGE_C_TIMESTEPS,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=seed_gen,
        callback=callback_prior,
    )
    images = wuerst_decoder_pipeline(
        image_embeddings=prior.image_embeddings,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        generator=seed_gen,
        output_type="pil",
    ).images

    if callback is not None:
        callback(steps, 0, 0, steps, np.array(images[0]))
        time.sleep(0.1)

    return images
