import gc
import numpy as np
import os
os.environ['HF_HOME'] = 'models/wuerstchen'

import torch
import warnings

import modules.core as core
import modules.path

from PIL import Image, ImageOps

from modules.settings import default_settings
from modules.util import suppress_stdout

import warnings
import time


#from diffusers import AutoPipelineForText2Image
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS



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
        wuerst_prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
            "warp-ai/wuerstchen-prior", torch_dtype=dtype).to(device)
        #wuerst_prior_pipeline.prior = torch.compile(wuerst_prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
    if wuerst_decoder_pipeline is None:
        wuerst_decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
            "warp-ai/wuerstchen", torch_dtype=dtype).to(device)
        #wuerst_decoder_pipeline.decoder = torch.compile(wuerst_decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)

#        wuerst_pipeline = AutoPipelineForText2Image.from_pretrained(
#            "warp-diffusion/wuerstchen", torch_dtype=dtype).to(device)

def load_refiner_model(model):
    return

def load_loras(loras):
    return

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

    if wuerst_prior_pipeline is None or wuerst_decoder_pipeline is None:
        load_base_model(None)

    seed_gen = torch.Generator()
    seed_gen.manual_seed(image_seed)

    prior = wuerst_prior_pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        height=round(height/128)*128,
        width=round(width/128)*128,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=seed_gen,
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

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return images
