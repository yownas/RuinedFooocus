import numpy as np
import os
import torch
import einops
import traceback
import cv2

import modules.async_worker as worker
from modules.util import generate_temp_filename
from PIL import Image

import os
from comfy.model_base import WAN21
import shared
from shared import path_manager, settings

from pathlib import Path
import random
from modules.pipleline_utils import (
    clean_prompt_cond_caches,
    get_previewer,
)

import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config

from calcuis_gguf.pig import load_gguf_sd, GGMLOps, GGUFModelPatcher

from nodes import (
    CLIPTextEncode,
    VAEDecodeTiled,
)
from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo
from comfy_extras.nodes_wan import WanImageToVideo 
from comfy_extras.nodes_model_advanced import ModelSamplingSD3


class pipeline:
    pipeline_type = ["wan_video"]

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
    model_hash_patched = ""
    model_base_patched = None
    conditions = None

    ggml_ops = GGMLOps()

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = 1 + ((int(gen_data["image_number"] / 4.0) + 1) * 4)
        gen_data["image_number"] = 1
        return gen_data

    def load_base_model(self, name, unet_only=True, hash=None): # Wan_Video never has the clip and vae models?
        # Check if model is already loaded
        if self.model_hash == name:
            return

        self.model_base = None
        self.model_hash = ""
        self.model_base_patched = None
        self.model_hash_patched = ""
        self.conditions = None

# FIXME? Add default model for video
#        default_name = path_manager.get_folder_file_path(
#            "checkpoints",
#            settings.default_settings.get("base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
#        )
#        default = shared.models.get_file("checkpoints", default_name)
        default = None

        filename = str(
            shared.models.get_model_path(
                "checkpoints",
                name,
                hash=hash,
                default=default,
            )
        )

        print(f"Loading WAN video {'unet' if unet_only else 'model'}: {name}")

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

                    if isinstance(unet.model, WAN21):
                        clip_name = settings.default_settings.get("clip_umt5", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
                        clip_names.append(str(clip_name))
                        clip_path = path_manager.get_folder_file_path(
                            "clip",
                            clip_name,
                            default = os.path.join(path_manager.model_paths["clip_path"], clip_name)
                        )
                        clip_paths.append(str(clip_path))
                        clip_type = comfy.sd.CLIPType.WAN
                        # https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
                        vae_name = settings.default_settings.get("vae_wan", "wan_2.1_vae.safetensors")

                    else:
                        print(f"ERROR: Not a Wan Video model?")
                        unet = None
                        return

                    print(f"Loading CLIP: {clip_names}")
                    clip = comfy.sd.load_clip(ckpt_paths=clip_paths, clip_type=clip_type, model_options={})

                    vae_path = path_manager.get_folder_file_path(
                        "vae",
                        vae_name,
                        default = os.path.join(path_manager.model_paths["vae_path"], vae_name)
                    )
                    print(f"Loading VAE: {vae_name}")
                    sd = comfy.utils.load_torch_file(str(vae_path))
                    vae = comfy.sd.VAE(sd=sd)

                    clip_vision_name = settings.default_settings.get("clip_vision", "clip_vision_h_fp8_e4m3fn.safetensors")
                    clip_vision_path = path_manager.get_folder_file_path(
                        "clip_vision",
                        clip_vision_name,
                        default = os.path.join(path_manager.model_paths["clip_vision_path"], clip_vision_name)
                    )
                    print(f"Loading CLIP Vision: {clip_vision_name}")
                    sd = comfy.utils.load_torch_file(str(clip_vision_path))
                    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd:
                        clip_vision = comfy.clip_vision.load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
                    else:
                        clip_vision = comfy.clip_vision.load_clipvision_from_sd(sd=sd)
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
                isinstance(self.model_base.unet.model, WAN21)
            ):
                print(
                    f"Model {type(self.model_base.unet.model)} not supported. Expected Wan Video model."
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
        loaded_loras = []

        model = self.model_base
        for lora in loras:
            name = lora.get("name", "None")
            weight = lora.get("weight", 0)
            hash = lora.get("hash", None)
            if name == "None" or weight == 0:
                continue

            filename = shared.models.get_model_path(
                "loras",
                name,
                hash=hash,
            )

            if filename is None:
                continue

            print(f"Loading LoRAs: {name}")
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
        self.model_base_patched = model
        self.model_hash_patched = str(loras)

        print(f"LoRAs loaded: {loaded_loras}")

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
                clip=self.model_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = hash
        update = True
        return update

    @torch.no_grad()
    def vae_decode_fake(self, latents):
        # FIXME: This should probably just be import from comfyui
        latent_rgb_factors = [
            [-0.1299, -0.1692,  0.2932],
            [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747],
            [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328],
            [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925],
            [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364],
            [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128],
            [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699],
            [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950],
            [ 0.1984,  0.0913,  0.1861]
        ]

        latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

        weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
        bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
        images = images.clamp(0.0, 1.0)

        return images

    @torch.inference_mode()
    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        shared.state["preview_total"] = 1

        seed = gen_data["seed"] if isinstance(gen_data["seed"], int) else random.randint(1, 2**32)

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"Processing text encoding ...", "html/generate_video.jpeg")
            )

        if self.conditions is None:
            self.conditions = clean_prompt_cond_caches()

        positive_prompt = gen_data["positive_prompt"]
        negative_prompt = gen_data["negative_prompt"]
        clip_skip = 1

        self.textencode("+", positive_prompt, clip_skip)
        self.textencode("-", negative_prompt, clip_skip)

        pbar = comfy.utils.ProgressBar(gen_data["steps"])

        def callback_function(step, x0, x, total_steps):
            y = self.vae_decode_fake(x0)
            y = (y * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            y = einops.rearrange(y, 'b c t h w -> (b h) (t w) c')
            # Skip callback() since we'll just confuse the preview grid and push updates outselves
            status = "Generating video"

            maxw = 1920
            maxh = 1080
            image = Image.fromarray(y)
            ow, oh = image.size
            scale = min(maxh / oh, maxw / ow)
            image = image.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)

            worker.add_result(
                gen_data["task_id"],
                "preview",
                (
                    int(100 * (step / total_steps)),
                    f"{status} - {step}/{total_steps}",
                    image
                )
            )
#            pbar.update_absolute(step + 1, total_steps, None)

        # ModelSamplingSD3
        model_sampling = ModelSamplingSD3().patch(
            model = self.model_base_patched.unet,
            shift = 8.0,
        )[0]

        # t2v or i2v?
        if gen_data["input_image"]:
            image = np.array(gen_data["input_image"]).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            clip_vision_output = self.model_base_patched.clip_vision.encode_image(image)

            (positive, negative, latent_image) = WanImageToVideo().encode(
                positive = self.conditions["+"]["cache"],
                negative = self.conditions["-"]["cache"],
                vae = self.model_base_patched.vae,
                width = gen_data["width"],
                height = gen_data["height"],
                length = gen_data["original_image_number"],
                batch_size = 1,
                start_image = image,
                clip_vision_output = clip_vision_output,
            )
        else:
            # latent_image
            latent_image = EmptyHunyuanLatentVideo().generate(
                width = gen_data["width"],
                height = gen_data["height"],
                length = gen_data["original_image_number"],
                batch_size = 1,
            )[0]
            positive = self.conditions["+"]["cache"]
            negative = self.conditions["-"]["cache"]

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", "html/generate_video.jpeg")
        )

        noise = comfy.sample.prepare_noise(latent_image["samples"], seed)

        sampled = comfy.sample.sample(
            model = model_sampling,
            noise = noise,
            steps = gen_data["steps"],
            cfg = gen_data["cfg"],
            sampler_name = gen_data["sampler_name"],
            scheduler = gen_data["scheduler"],
            positive = positive,
            negative = negative,
            latent_image = latent_image["samples"],

            denoise = 1,
            callback = callback_function,
        )

        if callback is not None:
            worker.add_result(
                gen_data["task_id"],
                "preview",
                (-1, f"VAE Decoding ...", None)
            )

        latent_image["samples"] = sampled

        decoded_latent = VAEDecodeTiled().decode(
            samples=latent_image,
            tile_size=128,
            overlap=64,
            vae=self.model_base_patched.vae,
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
        compress_level=9 # Min = 0, Max = 9

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
