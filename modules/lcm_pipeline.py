import gc
import numpy as np
import os
import warnings
from pathlib import Path

import modules.path
import modules.controlnet
import modules.async_worker as worker

from modules.settings import default_settings
from modules.util import suppress_stdout

import warnings
from diffusers import DiffusionPipeline
#from diffusers import LCMScheduler, LatentConsistencyModelPipeline
import torch

class pipeline():
    pipeline_type = ["lcm"]

    model_hash = ""
    pipe = None

    warnings.filterwarnings("ignore", category=UserWarning)

    def load_base_model(self, name):
        if self.model_hash == name:
            return

        filename = os.path.join(modules.path.modelfile_path, name)

        # ?
        #if self.xl_base is not None:
        #    self.xl_base.to_meta()
        #    self.xl_base = None

        # This is the only supported model at the moment
        model_id = "SimianLuo/LCM_Dreamshaper_v7"

        print(f"Loading model: {model_id}")

        #try:
        if True:
            def or_nice(image, device, dtype):
                return image, None

            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                local_files_only=False,
                use_safetensors=True,
                custom_pipeline="latent_consistency_txt2img",
                custom_revision="main",
            )
            
            #scheduler = LCMScheduler.from_pretrained(
            #    "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
            #self.pipe = LatentConsistencyModelPipeline.from_pretrained(
            #    "SimianLuo/LCM_Dreamshaper_v7", scheduler = scheduler, safety_checker = None)

            self.pipe.to(torch_device="cuda", torch_dtype=torch.float32)
            self.pipe.run_safety_checker = or_nice

            if self.pipe is not None:
                self.model_hash = name
                print(f"Base model loaded: {self.model_hash}")

        #except:
        #    print(f"Failed to load {name}")
        #    exit

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


    @torch.inference_mode()
    def process(
        self,
        positive_prompt,
        negative_prompt,
        input_image,
        controlnet,
        progress_window,
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
        worker.outputs.append(["preview", (-1, f"Generating ...", None)])
        
        torch.manual_seed(image_seed)

        images = self.pipe(
            prompt=positive_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            lcm_origin_steps=50,
            output_type="pil",
            width=width,
            height=height,
        ).images[0]

        if callback is not None:
            callback(steps, 0, 0, steps, images)

        gc.collect()

        return [images]
