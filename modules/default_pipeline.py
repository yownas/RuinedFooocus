import gc
import numpy as np
import os
import torch
import warnings

import modules.core as core
import modules.path

from PIL import Image, ImageOps

from comfy.model_base import SDXL, SDXLRefiner
from modules.settings import default_settings
from modules.util import suppress_stdout

import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)


xl_base: core.StableDiffusionModel = None
xl_base_hash = ""

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ""

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ""


def load_base_model(name):
    global xl_base, xl_base_hash, xl_base_patched, xl_base_patched_hash

    if xl_base_hash == name:
        return

    filename = os.path.join(modules.path.modelfile_path, name)

    if xl_base is not None:
        xl_base.to_meta()
        xl_base = None

    print(f"Loading base model: {name}")

    try:
        with suppress_stdout():
            xl_base = core.load_model(filename)

        if not isinstance(xl_base.unet.model, SDXL):
            print("Model not supported. Fooocus only support SDXL model as the base model.")
            xl_base = None

    except:
        print(f"Failed to load {name}, loading default model instead")
        load_base_model(modules.path.default_base_model_name)

    if xl_base is not None:
        xl_base_hash = name
        xl_base_patched = xl_base
        xl_base_patched_hash = ""
        print(f"Base model loaded: {xl_base_hash}")

    return


def load_refiner_model(name):
    global xl_refiner, xl_refiner_hash
    if xl_refiner_hash == str(name):
        return

    if name == "None":
        xl_refiner = None
        xl_refiner_hash = ""
        return

    filename = os.path.join(modules.path.modelfile_path, name)

    if xl_refiner is not None:
        xl_refiner.to_meta()
        xl_refiner = None

    print(f"Loading refiner model: {name}")
    with suppress_stdout():
        xl_refiner = core.load_model(filename)
    if not isinstance(xl_refiner.unet.model, SDXLRefiner):
        print("Model not supported. Fooocus only support SDXL refiner as the refiner.")
        xl_refiner = None
        xl_refiner_hash = ""
        print(f"Refiner unloaded.")
        return

    xl_refiner_hash = name
    print(f"Refiner model loaded: {xl_refiner_hash}")

    xl_refiner.vae.first_stage_model.to("meta")
    xl_refiner.vae = None
    return


def load_loras(loras):
    global xl_base, xl_base_patched, xl_base_patched_hash
    if xl_base_patched_hash == str(loras):
        return

    model = xl_base
    for name, weight in loras:
        if name == "None":
            continue

        filename = os.path.join(modules.path.lorafile_path, name)
        print(f"Loading LoRAs: {name}")
        with suppress_stdout():
            model = core.load_lora(model, filename, strength_model=weight, strength_clip=weight)
    xl_base_patched = model
    xl_base_patched_hash = str(loras)
    print(f"LoRAs loaded: {xl_base_patched_hash}")

    return


load_base_model(default_settings["base_model"])

positive_conditions_cache = None
negative_conditions_cache = None
positive_conditions_refiner_cache = None
negative_conditions_refiner_cache = None


def clean_prompt_cond_caches():
    global positive_conditions_cache, negative_conditions_cache, positive_conditions_refiner_cache, negative_conditions_refiner_cache
    positive_conditions_cache = None
    negative_conditions_cache = None
    positive_conditions_refiner_cache = None
    negative_conditions_refiner_cache = None
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
    global positive_conditions_cache, negative_conditions_cache, positive_conditions_refiner_cache, negative_conditions_refiner_cache

    with suppress_stdout():
        positive_conditions_cache = (
            core.encode_prompt_condition(clip=xl_base_patched.clip, prompt=positive_prompt)
            if positive_conditions_cache is None
            else positive_conditions_cache
        )
        negative_conditions_cache = (
            core.encode_prompt_condition(clip=xl_base_patched.clip, prompt=negative_prompt)
            if negative_conditions_cache is None
            else negative_conditions_cache
        )

    latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    force_full_denoise = True
    denoise = None

    if xl_refiner is not None:
        with suppress_stdout():
            positive_conditions_refiner_cache = (
                core.encode_prompt_condition(clip=xl_refiner.clip, prompt=positive_prompt)
                if positive_conditions_refiner_cache is None
                else positive_conditions_refiner_cache
            )
            negative_conditions_refiner_cache = (
                core.encode_prompt_condition(clip=xl_refiner.clip, prompt=negative_prompt)
                if negative_conditions_refiner_cache is None
                else negative_conditions_refiner_cache
            )

    sampled_latent = core.ksampler_with_refiner(
        model=xl_base_patched.unet,
        positive=positive_conditions_cache,
        negative=negative_conditions_cache,
        refiner=xl_refiner.unet if xl_refiner is not None else None,
        refiner_positive=positive_conditions_refiner_cache,
        refiner_negative=negative_conditions_refiner_cache,
        refiner_switch_step=switch,
        latent=latent,
        steps=steps,
        start_step=start_step,
        last_step=steps,
        disable_noise=False,
        force_full_denoise=force_full_denoise,
        denoise=denoise,
        seed=image_seed,
        sampler_name=sampler_name,
        scheduler=scheduler,
        cfg=cfg,
        callback_function=callback,
    )

    decoded_latent = core.decode_vae(vae=xl_base_patched.vae, latent_image=sampled_latent)

    images = core.image_to_numpy(decoded_latent)

    if callback is not None:
        callback(steps, 0, 0, steps, images[0])
        time.sleep(0.1)

    gc.collect()

    return images
