import modules.async_worker as worker

import os
import cv2
import imageio
import numpy as np
import rembg
import torch
import PIL.Image
from PIL import Image
from typing import Any


class pipeline:
    def remove_background(
        self,
        image: PIL.Image.Image,
        rembg_session: Any = None,
        force: bool = False,
        **rembg_kwargs,
    ) -> PIL.Image.Image:
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
        return image

    pipeline_type = ["rembg"]
    model_hash = ""

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        return

    def load_keywords(self, lora):
        return " "

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):

        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Removing background ...", None)
        )

        if gen_data["input_image"] is None:
            print(f"ERROR: Could not find input image.")
            return ["html/error.png"]

        rembg_session = rembg.new_session()
        image = self.remove_background(gen_data["input_image"], rembg_session)

        # Return finished image to preview
        if callback is not None:
            callback(1, 0, 0, 1, image)

        return [image]
