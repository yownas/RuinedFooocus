import gc
import numpy as np
import os
import torch
import cv2
import re

import modules.controlnet
import modules.async_worker as worker
import modules.prompt_processing as pp

from PIL import Image, ImageOps

from comfy.model_base import SDXL
from modules.settings import default_settings
from shared import path_manager

import time
import random

import einops
import comfy.utils
import comfy.model_management
from comfy.sd import load_checkpoint_guess_config
from tqdm import tqdm

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
from comfy_extras.nodes_freelunch import FreeU
from comfy.model_patcher import ModelPatcher
from comfy.utils import load_torch_file

from typing import Optional, Tuple
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
import torch.nn as nn


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# From https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/lib_layerdiffusion/models.py#L61
# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.latent_conv_in = zero_module(
            nn.Conv2d(4, block_out_channels[2], kernel_size=1)
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else output_channel
                ),
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=(
                attention_head_dim
                if attention_head_dim is not None
                else block_out_channels[-1]
            ),
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=(
                    attention_head_dim
                    if attention_head_dim is not None
                    else output_channel
                ),
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent

            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


# From https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/lib_layerdiffusion/models.py#L248
class TransparentVAEDecoder:
    def __init__(self, sd, device, dtype):
        self.load_device = device
        self.dtype = dtype

        model = UNet1024(in_channels=3, out_channels=4)
        model.load_state_dict(sd, strict=True)
        model.to(self.load_device, dtype=self.dtype)
        model.eval()
        self.model = model

    @torch.no_grad()
    def estimate_single_pass(self, pixel, latent):
        y = self.model(pixel, latent)
        return y

    @torch.no_grad()
    def estimate_augmented(self, pixel, latent):
        args = [
            [False, 0],
            [False, 1],
            [False, 2],
            [False, 3],
            [True, 0],
            [True, 1],
            [True, 2],
            [True, 3],
        ]

        result = []

        for flip, rok in tqdm(args):
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))

            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))

            eps = self.estimate_single_pass(feed_pixel, feed_latent).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))

            if flip:
                eps = torch.flip(eps, dims=(3,))

            result += [eps]

        result = torch.stack(result, dim=0)
        median = torch.median(result, dim=0).values
        return median

    @torch.no_grad()
    def decode_pixel(
        self, pixel: torch.TensorType, latent: torch.TensorType
    ) -> torch.TensorType:
        # pixel.shape = [B, C=3, H, W]
        assert pixel.shape[1] == 3
        pixel_device = pixel.device
        pixel_dtype = pixel.dtype

        pixel = pixel.to(device=self.load_device, dtype=self.dtype)
        latent = latent.to(device=self.load_device, dtype=self.dtype)
        # y.shape = [B, C=4, H, W]
        y = self.estimate_augmented(pixel, latent)
        y = y.clip(0, 1)
        assert y.shape[1] == 4
        # Restore image to original device of input image.
        return y.to(pixel_device, dtype=pixel_dtype)


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

    xl_base: StableDiffusionModel = None
    xl_base_hash = ""

    xl_base_patched: StableDiffusionModel = None
    xl_base_patched_hash = ""

    xl_controlnet: StableDiffusionModel = None
    xl_controlnet_hash = ""

    models = []
    inference_memory = None

    def load_base_model(self, name):
        if self.xl_base_hash == name:
            return

        filename = os.path.join(path_manager.model_paths["modelfile_path"], name)

        print(f"Loading base model: {name}")

        self.xl_base_patched = None
        self.xl_base_patched_hash = ""

        try:
            with torch.torch.inference_mode():
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
            self.load_base_model(
                path_manager.default_model_names["default_base_model_name"]
            )

        return

    def load_all_keywords(self, loras):
        lora_prompt_addition = ""
        return lora_prompt_addition

    def freeu(self, model, b1, b2, s1, s2):
        freeu_model = FreeU()
        unet = freeu_model.patch(model=model.unet, b1=b1, b2=b2, s1=s1, s2=s2)[0]
        return self.StableDiffusionModel(
            unet=unet, clip=model.clip, vae=model.vae, clip_vision=model.clip_vision
        )

    def load_loras(self, loras):
        lora_prompt_addition = self.load_all_keywords(loras)
        if self.xl_base_patched_hash == str(loras):
            return lora_prompt_addition

        loaded_loras = []

        model = self.xl_base
        for name, weight in loras:
            if name == "None" or weight == 0:
                continue
            filename = os.path.join(path_manager.model_paths["lorafile_path"], name)
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
        self.xl_base_patched = model
        # Uncomment below to enable FreeU shit
        # self.xl_base_patched = self.freeu(model, 1.01, 1.02, 0.99, 0.95)
        # self.xl_base_patched_hash = str(loras + [1.01, 1.02, 0.99, 0.95])
        self.xl_base_patched_hash = str(loras)

        print(f"LoRAs loaded: {loaded_loras}")

        return lora_prompt_addition

    def refresh_controlnet(self, name=None):
        if self.xl_controlnet_hash == str(self.xl_controlnet):
            return

        name = modules.controlnet.get_model(name)

        if name is not None and self.xl_controlnet_hash != name:
            filename = os.path.join(path_manager.model_paths["controlnet_path"], name)
            self.xl_controlnet = comfy.controlnet.load_controlnet(filename)
            self.xl_controlnet_hash = name
            print(f"ControlNet model loaded: {self.xl_controlnet_hash}")
        if self.xl_controlnet_hash != name:
            self.xl_controlnet = None
            self.xl_controlnet_hash = None
            print(f"Controlnet model unloaded")

    conditions = None

    def clean_prompt_cond_caches(self):
        self.conditions = {}
        self.conditions["+"] = {}
        self.conditions["-"] = {}
        self.conditions["switch"] = {}
        self.conditions["+"]["text"] = None
        self.conditions["+"]["cache"] = None
        self.conditions["-"]["text"] = None
        self.conditions["-"]["cache"] = None
        self.conditions["switch"]["text"] = None
        self.conditions["switch"]["cache"] = None

    def textencode(self, id, text):
        update = False
        if text != self.conditions[id]["text"]:
            self.conditions[id]["cache"] = CLIPTextEncode().encode(
                clip=self.xl_base_patched.clip, text=text
            )[0]
        self.conditions[id]["text"] = text
        update = True
        return update

    def set_timestep_range(self, conditioning, start, end):
        c = []
        for t in conditioning:
            if "pooled_output" in t:
                t["start_percent"] = start
                t["end_percent"] = end

        return conditioning

    # From https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/lib_layerdiffusion/utils.py#L118
    def to_lora_patch_dict(self, state_dict: dict) -> dict:
        """Convert raw lora state_dict to patch_dict that can be applied on
        modelpatcher."""
        patch_dict = {}
        for k, w in state_dict.items():
            model_key, patch_type, weight_index = k.split("::")
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        return patch_flat

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
        try:
            if self.xl_base_patched == None or not isinstance(
                self.xl_base_patched.unet.model, SDXL
            ):
                print(f"ERROR: Can not use old 1.5 model")
                worker.interrupt_ruined_processing = True
                worker.outputs.append(
                    ["preview", (-1, f"Can not use old 1.5 model ...", "error.png")]
                )
                return []
        except Exception as e:
            # Something went very wrong
            print(f"ERROR: {e}")
            worker.interrupt_ruined_processing = True
            worker.outputs.append(
                ["preview", (-1, f"Error when trying to use model ...", "error.png")]
            )
            return []

        img2img_mode = False
        layerdiffusion_mode = False
        seed = image_seed if isinstance(image_seed, int) else random.randint(1, 2**32)

        worker.outputs.append(["preview", (-1, f"Processing text encoding ...", None)])
        updated_conditions = False
        if self.conditions is None:
            self.clean_prompt_cond_caches()

        if self.textencode("+", positive_prompt):
            updated_conditions = True
        if self.textencode("-", negative_prompt):
            updated_conditions = True

        prompt_switch_mode = False
        if "[" in positive_prompt and "]" in positive_prompt:
            prompt_switch_mode = True

        if prompt_switch_mode and controlnet is not None and input_image is not None:
            print(
                "ControlNet and [prompt|switching] do not work well together. ControlNet will be applied to the first prompt only."
            )

        if prompt_switch_mode:
            prompt_switch_mode = True
            prompt_per_step = pp.prompt_switch_per_step(positive_prompt, steps)

            perc_per_step = round(100 / steps, 2)
            positive_complete = []
            for i in range(len(prompt_per_step)):
                if self.textencode("switch", prompt_per_step[i]):
                    updated_conditions = True
                positive_switch = convert_cond(self.conditions["switch"]["cache"])
                start_perc = round((perc_per_step * i) / 100, 2)
                end_perc = round((perc_per_step * (i + 1)) / 100, 2)
                if end_perc >= 0.99:
                    end_perc = 1
                positive_switch = self.set_timestep_range(
                    positive_switch, start_perc, end_perc
                )

                positive_complete += positive_switch

            positive_switch = convert_cond(self.conditions["switch"]["cache"])

        device = comfy.model_management.get_torch_device()

        if controlnet is not None and "type" in controlnet and input_image is not None:
            worker.outputs.append(["preview", (-1, f"Powering up ...", None)])
            input_image = input_image.convert("RGB")
            input_image = np.array(input_image).astype(np.float32) / 255.0
            input_image = torch.from_numpy(input_image)[None,]
            input_image = ImageScaleToTotalPixels().upscale(
                image=input_image, upscale_method="bicubic", megapixels=1.0
            )[0]
            self.refresh_controlnet(name=controlnet["type"])
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
            if self.xl_controlnet:
                if prompt_switch_mode:
                    (
                        self.conditions["+"]["cache"],
                        self.conditions["-"]["cache"],
                    ) = ControlNetApplyAdvanced().apply_controlnet(
                        positive=positive_complete,
                        negative=self.conditions["-"]["cache"],
                        control_net=self.xl_controlnet,
                        image=input_image,
                        strength=float(controlnet["strength"]),
                        start_percent=float(controlnet["start"]),
                        end_percent=float(controlnet["stop"]),
                    )
                    self.conditions["+"]["text"] = None
                    self.conditions["-"]["text"] = None
                else:
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

        if controlnet is not None and "type" in controlnet:
            if controlnet["type"].lower() == "layerdiffusion":
                tmodel = ModelPatcher(self.xl_base_patched.unet, device, "cpu", size=1)
                layer_lora_state_dict = load_torch_file(
                    "models/layerdiffuse/layer_xl_transparent_attn.safetensors"
                )
                layer_lora_patch_dict = self.to_lora_patch_dict(layer_lora_state_dict)
                # weight = 1.0
                tmodel.model.add_patches(layer_lora_patch_dict)
                self.xl_base_patched.unet = tmodel.model
                self.xl_base_patched_hash = ""

                # load transparent vae
                self.xl_base_patched.tvae = TransparentVAEDecoder(
                    load_torch_file(
                        "models/layerdiffuse/vae_transparent_decoder.safetensors"
                    ),
                    device=comfy.model_management.get_torch_device(),
                    dtype=(
                        torch.float16
                        if comfy.model_management.should_use_fp16()
                        else torch.float32
                    ),
                )
                layerdiffusion_mode = True

        if not img2img_mode:
            latent = EmptyLatentImage().generate(
                width=width, height=height, batch_size=1
            )[0]
            force_full_denoise = True
            denoise = None

        if gen_data["inpaint_toggle"]:
            mask = gen_data["inpaint_view"]["mask"]
            mask = mask[:, :, 0]
            mask = torch.from_numpy(mask)[None,] / 255.0

            image = gen_data["inpaint_view"]["image"]
            image = image[..., :-1]
            image = torch.from_numpy(image)[None,] / 255.0

            latent = VAEEncodeForInpaint().encode(
                vae=self.xl_base_patched.vae,
                pixels=image,
                mask=mask,
                grow_mask_by=20,
            )[0]

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

        comfy.model_management.load_models_gpu([self.xl_base_patched.unet])
        comfy.model_management.load_models_gpu(self.models)

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        if prompt_switch_mode:
            positive_copy = positive_complete
        else:
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

        if layerdiffusion_mode:
            pixel = decoded_latent[0].permute(2, 0, 1).unsqueeze(0)

            ## Decoder requires dimension to be 64-aligned.
            B, C, H, W = pixel.shape
            assert H % 64 == 0, f"Height({H}) is not multiple of 64."
            assert W % 64 == 0, f"Height({W}) is not multiple of 64."

            decoded = []
            sub_batch_size = 1
            for start_idx in range(0, samples.shape[0], sub_batch_size):
                decoded.append(
                    self.xl_base_patched.tvae.decode_pixel(
                        pixel[start_idx : start_idx + sub_batch_size],
                        samples[start_idx : start_idx + sub_batch_size],
                    )
                )
            pixel_with_alpha = torch.cat(decoded, dim=0)

            # [B, C, H, W] => [B, H, W, C]
            pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
            image = pixel_with_alpha[..., 1:]
            alpha = pixel_with_alpha[..., 0]

            i= np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
            i= np.squeeze(i)
            a= np.clip(255.0 * alpha[0].cpu().numpy(), 0, 255).astype(np.uint8)
            a= np.squeeze(a)
            img = Image.fromarray(i).convert("RGBA")
            img.putalpha(Image.fromarray(a).convert("L"))

            images = [img]

        if callback is not None:
            callback(steps, 0, 0, steps, images[0])

        return images
