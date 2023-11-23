import gc
import numpy as np
import os
import torch

import modules.path
import modules.controlnet
import modules.async_worker as worker

from PIL import Image, ImageOps

from comfy.model_base import SDXL
from modules.settings import default_settings
from modules.util import suppress_stdout

import time
import random

import einops
import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config
from comfy_extras.chainner_models import model_loading
from nodes import (
    CLIPTextEncode,
    ControlNetApplyAdvanced,
    EmptyLatentImage,
    VAEDecode,
    VAEEncode,
    VAEEncodeForInpaint,
)
from comfy.sample import (
    cleanup_additional_models,
    convert_cond,
    get_additional_models,
    prepare_mask,
)
from comfy.samplers import KSampler
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel


class pipeline:
    pipeline_type = ["sdxl", "ssd"]

    comfy.model_management.DISABLE_SMART_MEMORY = True

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

    def get_previewer(self, device, latent_format):
        from latent_preview import TAESD, TAESDPreviewerImpl

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
            with torch.no_grad():
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

    xl_base: StableDiffusionModel = None
    xl_base_hash = ""

    xl_base_patched: StableDiffusionModel = None
    xl_base_patched_hash = ""

    xl_controlnet: StableDiffusionModel = None
    xl_controlnet_hash = ""

    models = []
    inference_memory = None

    def load_upscaler_model(self, model_name):
        model_path = os.path.join(modules.path.upscaler_path, model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = model_loading.load_state_dict(sd).eval()
        return out

    def load_base_model(self, name):
        if self.xl_base_hash == name:
            return

        filename = os.path.join(modules.path.modelfile_path, name)

        print(f"Loading base model: {name}")

        try:
            with suppress_stdout():
                unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)
                self.xl_base = self.StableDiffusionModel(
                    unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
                )
            if not isinstance(self.xl_base.unet.model, SDXL):
                print(
                    "Model not supported. Fooocus only support SDXL model as the base model."
                )
                self.xl_base = None

            if self.xl_base is not None:
                self.xl_base_hash = name
                self.xl_base_patched = self.xl_base
                self.xl_base_patched_hash = ""
                # self.xl_base_patched.unet.model.to("cuda")
                print(f"Base model loaded: {self.xl_base_hash}")

        except:
            print(f"Failed to load {name}, loading default model instead")
            load_base_model(modules.path.default_base_model_name)

        return

    def load_keywords(self, lora):
        filename = lora.replace(".safetensors", ".txt")
        try:
            with open(filename, "r") as file:
                data = file.read()
            return data
        except FileNotFoundError:
            return " "

    def load_loras(self, loras):
        if self.xl_base_patched_hash == str(loras):
            return

        lora_prompt_addition = ""
        loaded_loras = []

        model = self.xl_base
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
                    model = self.StableDiffusionModel(
                        unet=unet,
                        clip=clip,
                        vae=model.vae,
                        clip_vision=model.clip_vision,
                    )
                    loaded_loras += [(name, weight)]
                except:
                    pass
                lora_prompt_addition = (
                    f"{lora_prompt_addition}, {self.load_keywords(filename)}"
                )
        self.xl_base_patched = model
        self.xl_base_patched_hash = str(loras)
        print(f"LoRAs loaded: {loaded_loras}")

        return lora_prompt_addition

    def refresh_controlnet(self, name=None):
        if self.xl_controlnet_hash == str(self.xl_controlnet):
            return

        name = modules.controlnet.get_model(name)

        if name is not None and self.xl_controlnet_hash != name:
            filename = os.path.join(modules.path.controlnet_path, name)
            self.xl_controlnet = comfy.controlnet.load_controlnet(filename)
            self.xl_controlnet_hash = name
            print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
        return

    conditions = None

    def clean_prompt_cond_caches(self):
        self.conditions = {}
        self.conditions["+"] = {}
        self.conditions["-"] = {}
        self.conditions["+"]["text"] = None
        self.conditions["+"]["cache"] = None
        self.conditions["-"]["text"] = None
        self.conditions["-"]["cache"] = None

    def textencode(self, id, text):
        update = False
        if text != self.conditions[id]["text"]:
            with suppress_stdout():
                self.conditions[id]["cache"] = CLIPTextEncode().encode(
                    clip=self.xl_base_patched.clip, text=text
                )[0]
            self.conditions[id]["text"] = text
            update = True
        return update

    @torch.inference_mode()
    def process(
        self,
        positive_prompt,
        negative_prompt,
        input_image,
        controlnet,
        main_view,
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
        gen_data=None,
    ):
        img2img_mode = False
        seed = image_seed if isinstance(image_seed, int) else random.randint(1, 2**32)

        worker.outputs.append(["preview", (-1, f"Processing text encoding ...", None)])
        updated_conditions = False
        if self.conditions is None:
            self.clean_prompt_cond_caches()
        if self.textencode("+", positive_prompt):
            updated_conditions = True
        if self.textencode("-", negative_prompt):
            updated_conditions = True

        if controlnet is not None and input_image is not None:
            worker.outputs.append(["preview", (-1, f"Powering up ...", None)])
            input_image = input_image.convert("RGB")
            input_image = np.array(input_image).astype(np.float32) / 255.0
            input_image = torch.from_numpy(input_image)[None,]
            input_image = ImageScaleToTotalPixels().upscale(
                image=input_image, upscale_method="bicubic", megapixels=1.0
            )[0]
            self.refresh_controlnet(name=controlnet["type"])
            if self.xl_controlnet:
                match controlnet["type"].lower():
                    case "canny":
                        input_image = Canny().detect_edge(
                            image=input_image,
                            low_threshold=float(controlnet["edge_low"]),
                            high_threshold=float(controlnet["edge_high"]),
                        )[0]
                        updated_conditions = True
                    case "depth":
                        updated_conditions = True
                (
                    self.conditions["+"]["cache"],
                    self.conditions["-"]["cache"],
                ) = ControlNetApplyAdvanced().apply_controlnet(
                    positive=self.conditions["+"]["cache"],
                    negative=self.conditions["-"]["cache"],
                    control_net=self.xl_controlnet,
                    image=input_image,
                    strength=float(controlnet["strength"]),
                    start_percent=float(controlnet["start"]),
                    end_percent=float(controlnet["stop"]),
                )
                self.conditions["+"]["text"] = None
                self.conditions["-"]["text"] = None

            if controlnet["type"].lower() == "img2img":
                latent = VAEEncode().encode(
                    vae=self.xl_base_patched.vae, pixels=input_image
                )[0]
                force_full_denoise = False
                denoise = float(controlnet.get("denoise", controlnet.get("strength")))
                img2img_mode = True
            if controlnet["type"].lower() == "upscale":
                worker.outputs.append(["preview", (-1, f"Upscaling image ...", None)])
                upscaler_model = self.load_upscaler_model(controlnet["upscaler"])
                decoded_latent = ImageUpscaleWithModel().upscale(
                    upscaler_model, input_image
                )[0]

                images = [
                    np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
                    for y in decoded_latent
                ]
                return images

        if not img2img_mode:
            latent = EmptyLatentImage().generate(
                width=width, height=height, batch_size=1
            )[0]
            force_full_denoise = True
            denoise = None

        if gen_data["inpaint_toggle"]:
            mask = gen_data["inpaint_view"]["mask"]
            # mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)[None,]
            image = gen_data["inpaint_view"]["image"]
            # image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            latent = VAEEncodeForInpaint().encode(
                vae=self.xl_base_patched.vae,
                pixels=image,
                mask=mask,
            )[0]

        device = comfy.model_management.get_torch_device()

        latent_image = latent["samples"]
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        previewer = self.get_previewer(
            device, self.xl_base_patched.unet.model.latent_format
        )

        pbar = comfy.utils.ProgressBar(steps)

        def callback_function(step, x0, x, total_steps):
            y = None
            if previewer:
                y = previewer.preview(x0, step, total_steps)
            if callback is not None:
                callback(step, x0, x, total_steps, y)
            pbar.update_absolute(step + 1, total_steps, None)

        if noise_mask is not None:
            noise_mask = prepare_mask(noise_mask, noise.shape, device)

        worker.outputs.append(["preview", (-1, f"Prepare models ...", None)])
        if updated_conditions:
            self.models, self.inference_memory = get_additional_models(
                self.conditions["+"]["cache"],
                self.conditions["-"]["cache"],
                self.xl_base_patched.unet.model_dtype(),
            )

        with suppress_stdout():
            comfy.model_management.load_models_gpu([self.xl_base_patched.unet])
            comfy.model_management.load_models_gpu(self.models)

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive_copy = convert_cond(self.conditions["+"]["cache"])
        negative_copy = convert_cond(self.conditions["-"]["cache"])
        kwargs = {
            "cfg": cfg,
            "latent_image": latent_image,
            "start_step": start_step,
            "last_step": steps,
            "force_full_denoise": force_full_denoise,
            "denoise_mask": noise_mask,
            "sigmas": None,
            "disable_pbar": False,
            "seed": seed,
        }
        sampler = KSampler(
            self.xl_base_patched.unet.model,
            steps=steps,
            device=device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            model_options=self.xl_base_patched.unet.model_options,
        )
        extra_kwargs = {
            "callback": callback_function,
        }
        kwargs.update(extra_kwargs)

        worker.outputs.append(["preview", (-1, f"Start sampling ...", None)])
        samples = sampler.sample(noise, positive_copy, negative_copy, **kwargs)

        samples = samples.cpu()

        cleanup_additional_models(self.models)

        sampled_latent = latent.copy()
        sampled_latent["samples"] = samples

        worker.outputs.append(["preview", (-1, f"VAE decoding ...", None)])

        decoded_latent = VAEDecode().decode(
            samples=sampled_latent, vae=self.xl_base_patched.vae
        )[0]

        images = [
            np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8)
            for y in decoded_latent
        ]

        if callback is not None:
            callback(steps, 0, 0, steps, images[0])
            time.sleep(0.1)

        gc.collect()

        return images
