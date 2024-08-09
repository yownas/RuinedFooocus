import modules.async_worker as worker

from modules.settings import default_settings
import torch
import diffusers
import re
import os

from PIL import Image
from shared import path_manager


class pipeline:
    pipeline_type = ["diffusers"]

    model_hash = ""
    pipe = None

#    def parse_gen_data(self, gen_data):
#        return gen_data

    def load_base_model(self, name):
        # Check if model is already loaded
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
        self.model_hash = name
        repo_id = re.sub(r"🤗:", "", name, count=1)

        # TODO get proper pipeline from model_config.json
        self.pipe = diffusers.FluxPipeline.from_pretrained(
            os.path.join(path_manager.model_paths["diffusers_path"], repo_id),
            local_files_only = True,
            torch_dtype=torch.bfloat16
            )
        self.pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
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
        clip_skip,
        callback,
        gen_data=None,
    ):
        worker.outputs.append(["preview", (-1, f"Generating ...", None)])

        seed = image_seed

        image = self.pipe(
            positive_prompt,
            height=height,
            width=width,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]

        # Return finished image to preview
        if callback is not None:
            callback(steps, 0, 0, steps, image)

        return [image]
