import numpy as np
import os
import torch
import traceback
import cv2

import modules.async_worker as worker
from modules.settings import default_settings
from modules.util import generate_temp_filename
from PIL import Image

import os
from comfy.model_base import BaseModel, SDXL, SD3, Flux, Lumina2, HunyuanVideo
from modules.settings import default_settings
from shared import path_manager

from pathlib import Path
import json
import random
from modules.pipleline_utils import (
    clean_prompt_cond_caches,
)

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config
from tqdm import tqdm

from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher
from calcuis_gguf.pig import DualClipLoaderGGUF as DualCLIPLoaderGGUF

from nodes import (
    CLIPTextEncode,
    DualCLIPLoader,
    VAEDecodeTiled,
)

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, RandomNoise, BasicScheduler, KSamplerSelect, BasicGuider
from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_images import SaveAnimatedPNG
from node_helpers import conditioning_set_values


class pipeline:
    pipeline_type = ["hunyuan_video"]

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

    model_hash = ""
    model_base = None
    conditions = None

    ggml_ops = GGMLOps()

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = 1 + ((int(gen_data["image_number"] / 4.0) + 1) * 4)
        gen_data["image_number"] = 1
        return gen_data

    def load_base_model(self, name, unet_only=True): # Hunyuan_Video never has the clip and vae models?
        # Check if model is already loaded
        if self.model_hash == name:
            return

        self.model_base = None
        self.model_hash = ""
        self.conditions = None

        filename = os.path.join(path_manager.model_paths["modelfile_path"], name)

        print(f"Loading base {'unet' if unet_only else 'model'}: {name}")

        if filename.endswith(".gguf") or unet_only:
            with torch.torch.inference_mode():
                try:
                    if filename.endswith(".gguf"):
                        sd = load_gguf_sd(filename)
                        unet = comfy.sd.load_diffusion_model_state_dict(
                            sd, model_options={"custom_operations": self.ggml_ops}
                        )
                        unet = GGUFModelPatcher.clone(unet)
                        unet.patch_on_device = True
                    else:
                        model_options = {}
                        model_options["dtype"] = torch.float8_e4m3fn # FIXME should be a setting
                        unet = comfy.sd.load_diffusion_model(filename, model_options=model_options)

                    clip_paths = []
                    clip_names = []

                    if isinstance(unet.model, HunyuanVideo):
                        clip_name = default_settings.get("clip_l", "clip_l.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))
                        # https://huggingface.co/calcuis/hunyuan-gguf/tree/main
                        clip_name = default_settings.get("clip_llava", "llava_llama3_fp8_scaled.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))
                        clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
                        # https://huggingface.co/calcuis/hunyuan-gguf/tree/main
                        vae_name = default_settings.get("vae_hunyuan_video", "hunyuan_video_vae_bf16.safetensors")

                    else:
                        print(f"ERROR: Not a Hunyuan Video model?")
                        unet = None
                        return

                    print(f"Loading CLIP: {clip_names}")
                    clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
                    clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=clip_type, model_options={})

                    vae_path = path_manager.get_folder_file_path(
                        "vae",
                        vae_name,
                        default = os.path.join(path_manager.model_paths["vae_path"], vae_name)
                    )
                    print(f"Loading VAE: {vae_name}")
                    sd = comfy.utils.load_torch_file(str(vae_path))
                    vae = comfy.sd.VAE(sd=sd)

                    clip_vision = None
                except Exception as e:
                    unet = None
                    traceback.print_exc() 

        else:
            try:
                with torch.torch.inference_mode():
                    unet, clip, vae, clip_vision = load_checkpoint_guess_config(filename)

                if clip == None or vae == None:
                    raise
            except:
                print(f"Failed. Trying to load as unet.")
                self.load_base_model(
                    filename,
                    unet_only=True
                )
                return

        if unet == None:
            print(f"Failed to load {name}")
            self.model_base = None
            self.model_hash = ""
        else:
            self.model_base = self.StableDiffusionModel(
                unet=unet, clip=clip, vae=vae, clip_vision=clip_vision
            )
            if not (
                isinstance(self.model_base.unet.model, HunyuanVideo)
            ):
                print(
                    f"Model {type(self.model_base.unet.model)} not supported. Expected Hunyuan Video model."
                )
                self.model_base = None

            if self.model_base is not None:
                self.model_hash = name
                print(f"Base model loaded: {self.model_hash}")
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
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    conditions = None

    def textencode(self, id, text, clip_skip):
        update = False
        hash = f"{text} {clip_skip}"
        if hash != self.conditions[id]["text"]:
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.model_base.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Processing text encoding ...", None)
            )
        updated_conditions = False
        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        clip_skip = 1

        if self.textencode("+", positive_prompt, clip_skip):
            updated_conditions = True
        if self.textencode("-", negative_prompt, clip_skip):
            updated_conditions = True

        conds = {
            "positive": self.conditions["+"]["cache"],
            "negative": self.conditions["-"]["cache"],
        }

        previewer = None

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            y = None
            if previewer:
                y = previewer.preview(x0, step, total_steps)
            if callback is not None:
                callback(step, x0, x, total_steps, y)
            pbar.update_absolute(step + 1, total_steps, None)

        # Noise
        noise = RandomNoise().get_noise(noise_seed=seed)[0]

        # Guider
        model_sampling = ModelSamplingSD3().patch(
            model = self.model_base.unet,
            shift = 7.0,
        )[0]
        flux_guideance = FluxGuidance().append(
            conditioning = self.conditions["+"]["cache"],
            guidance = gen_data["cfg"],
        )[0]

        guider = BasicGuider().get_guider(
            model = self.model_base.unet,
            conditioning = flux_guideance,
        )[0]

        # Sampler
        ksampler = KSamplerSelect().get_sampler(
            sampler_name = gen_data["sampler_name"],
        )[0]

        # Sigmas
        sigmas = BasicScheduler().get_sigmas(
            model = self.model_base.unet,
            scheduler = gen_data["scheduler"],
            steps = gen_data["steps"],
            denoise = 1,
        )[0]

        # latent_image
        latent_image = EmptyHunyuanLatentVideo().generate(
            width = gen_data["width"],
            height = gen_data["height"],
            length = gen_data["original_image_number"],
            batch_size = 1,
        )[0]

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", None)
        )

        sampled = SamplerCustomAdvanced().sample(
            noise = noise,
            guider = guider,
            sampler = ksampler,
            sigmas = sigmas,
            latent_image = latent_image,
        )[0]

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        decoded_latent = VAEDecodeTiled().decode(
            samples=sampled,
            tile_size=64,
            overlap=32,
            vae=self.model_base.vae,
        )[0]

        pil_images = []
        for image in decoded_latent:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Saving ...", None)
            )

        file = generate_temp_filename(
            folder=path_manager.model_paths["temp_outputs_path"], extension="gif"
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)

        fps=12.0
        compress_level=4 # Min = 0, Max = 9

        # Save GIF
        pil_images[0].save(
            file,
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0/fps),
            append_images=pil_images[1:],
            optimize=True,
            loop=0,
        )

        # Save mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mp4_file = file.with_suffix(".mp4")
        out = cv2.VideoWriter(mp4_file, fourcc, fps, (gen_data["width"], gen_data["height"]))
        for frame in pil_images:
            out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
        out.release()

        return [file]
