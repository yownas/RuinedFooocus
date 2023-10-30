import gc
import numpy as np
import os
import torch
import warnings

import modules.core as core
import modules.path
import modules.controlnet
import modules.async_worker as worker

from PIL import Image, ImageOps

from comfy.model_base import SDXL
from modules.settings import default_settings
from modules.util import suppress_stdout

import warnings
import time

import comfy.utils
from comfy.sd import load_checkpoint_guess_config
from nodes import (
    CLIPTextEncode,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEDecode,
    VAEEncode,
)
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny



warnings.filterwarnings("ignore", category=UserWarning)

class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision

    def to_meta(self):
        if self.unet is not None:
            self.unet.model.to("meta")
        if self.clip is not None:
            self.clip.cond_stage_model.to("meta")
        if self.vae is not None:
            self.vae.first_stage_model.to("meta")


xl_base: StableDiffusionModel = None
xl_base_hash = ""

xl_base_patched: StableDiffusionModel = None
xl_base_patched_hash = ""

xl_controlnet: StableDiffusionModel = None
xl_controlnet_hash = ""


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
            unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)
            xl_base = StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)
        if not isinstance(xl_base.unet.model, SDXL):
            print(
                "Model not supported. Fooocus only support SDXL model as the base model."
            )
            xl_base = None

        if xl_base is not None:
            xl_base_hash = name
            xl_base_patched = xl_base
            xl_base_patched_hash = ""
            print(f"Base model loaded: {xl_base_hash}")

    except:
        print(f"Failed to load {name}, loading default model instead")
        load_base_model(modules.path.default_base_model_name)

    return


def load_keywords(lora):
    filename = lora.replace(".safetensors", ".txt")
    try:
        with open(filename, "r") as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return " "


def load_loras(loras):
    global xl_base, xl_base_patched, xl_base_patched_hash
    if xl_base_patched_hash == str(loras):
        return

    lora_prompt_addition = ""

    model = xl_base
    for name, weight in loras:
        if name == "None" or weight == 0:
            continue

        filename = os.path.join(modules.path.lorafile_path, name)
        print(f"Loading LoRAs: {name}")
        with suppress_stdout():
            try:
                lora = comfy.utils.load_torch_file(filename, safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(
                    model.unet, model.clip, lora, weight, weight
                )
                model = StableDiffusionModel(
                    unet=unet, clip=clip, vae=model.vae, clip_vision=model.clip_vision
                )
            except:
                pass
            lora_prompt_addition = f"{lora_prompt_addition}, {load_keywords(filename)}"
    xl_base_patched = model
    xl_base_patched_hash = str(loras)
    print(f"LoRAs loaded: {xl_base_patched_hash}")

    return lora_prompt_addition


def refresh_controlnet(name=None):
    global xl_controlnet, xl_controlnet_hash
    if xl_controlnet_hash == str(xl_controlnet):
        return

    name = modules.controlnet.get_model(name)

    if name is not None and xl_controlnet_hash != name:
        filename = os.path.join(modules.path.controlnet_path, name)
        xl_controlnet = comfy.controlnet.load_controlnet(filename)
        xl_controlnet_hash = name
        print(f"ControlNet model loaded: {xl_controlnet_hash}")
    return


load_base_model(default_settings["base_model"])

positive_conditions_cache = None
negative_conditions_cache = None


def clean_prompt_cond_caches():
    global positive_conditions_cache, negative_conditions_cache
    positive_conditions_cache = None
    negative_conditions_cache = None
    return

@torch.inference_mode()
def process(
    positive_prompt,
    negative_prompt,
    input_image,
    controlnet,
    steps,
    width,
    height,
    image_seed,
    start_step,
    denoise,
    cfg,
    sampler_name,
    scheduler,
    callback,
):
    global positive_conditions_cache, negative_conditions_cache
    global xl_controlnet

    worker.outputs.append(["preview", (-1, f"Processing text encoding ...", None)])
    img2img_mode = False

    with suppress_stdout():
        positive_conditions_cache = (
            CLIPTextEncode().encode(
                clip=xl_base_patched.clip, text=positive_prompt
            )[0]
            if positive_conditions_cache is None
            else positive_conditions_cache
        )
        negative_conditions_cache = (
            CLIPTextEncode().encode(
                clip=xl_base_patched.clip, text=negative_prompt
            )[0]
            if negative_conditions_cache is None
            else negative_conditions_cache
        )

    if controlnet is not None and input_image is not None:
        input_image = input_image.convert("RGB")
        input_image = np.array(input_image).astype(np.float32) / 255.0
        input_image = torch.from_numpy(input_image)[None,]
        input_image = ImageScaleToTotalPixels().upscale(
            image=input_image, upscale_method="bicubic", megapixels=1.0
        )[0]
        refresh_controlnet(name=controlnet["type"])
        if xl_controlnet:
            match controlnet["type"].lower():
                case "canny":
                    input_image = Canny().detect_edge(
                        image=input_image,
                        low_threshold=float(controlnet["edge_low"]),
                        high_threshold=float(controlnet["edge_high"]),
                    )[0]
                # case "depth": (no preprocessing?)
            (
                positive_conditions_cache,
                negative_conditions_cache,
            ) = ControlNetApplyAdvanced().apply_controlnet(
                positive=positive_conditions_cache,
                negative=negative_conditions_cache,
                control_net=xl_controlnet,
                image=input_image,
                strength=float(controlnet["strength"]),
                start_percent=float(controlnet["start"]),
                end_percent=float(controlnet["stop"]),
            )

        if controlnet["type"].lower() == "img2img":
            latent = VAEEncode().encode(vae=xl_base_patched.vae, pixels=input_image)[0]
            force_full_denoise = False
            denoise = float(controlnet.get("denoise", controlnet.get("strength")))
            img2img_mode = True

    if not img2img_mode:
        latent = EmptyLatentImage().generate(
            width=width, height=height, batch_size=1
        )[0]
        force_full_denoise = True
        denoise = None

    worker.outputs.append(["preview", (-1, f"Start sampling ...", None)])

    sampled_latent = core.ksampler(
        model=xl_base_patched.unet,
        positive=positive_conditions_cache,
        negative=negative_conditions_cache,
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

    worker.outputs.append(["preview", (-1, f"VAE decoding ...", None)])

    decoded_latent = VAEDecode().decode(
        samples=sampled_latent, vae=xl_base_patched.vae
    )[0]

    images = [np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in decoded_latent]

    if callback is not None:
        callback(steps, 0, 0, steps, images[0])
        time.sleep(0.1)

    gc.collect()

    return images
