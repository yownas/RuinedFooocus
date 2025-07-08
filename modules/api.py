import time
import gradio as gr
from typing import Any
import base64
import re
import numpy as np
from pathlib import Path
import shared
from shared import (
    state,
    settings,
)
import modules.search_pipeline as search_pipeline

def add_api():

    # llama
    from modules.llama_pipeline import run_llama
    def api_llama(system: str, user: str) -> str:
        """
        Talk to the RuinedFooocus llama llm model.

        Args:
            system (str): System prompt
            user (str): user input
        """
        prompt = f"system: {system}\n\n{user}"
        return run_llama(None, prompt)
    gr.api(api_llama, api_name="llama")

    # process
    import modules.async_worker as worker
    def _api_process(prompt: str) -> list:
        tmp_data = {
            'task_type': "api_process",
            'prompt': prompt,
            'negative': "",
            'loras': [
                ("", f"{settings.default_settings.get('lora_1_weight', 1.0)} - {settings.default_settings.get('lora_1_model', 'None')}"),
                ("", f"{settings.default_settings.get('lora_2_weight', 1.0)} - {settings.default_settings.get('lora_2_model', 'None')}"),
                ("", f"{settings.default_settings.get('lora_3_weight', 1.0)} - {settings.default_settings.get('lora_3_model', 'None')}"),
                ("", f"{settings.default_settings.get('lora_4_weight', 1.0)} - {settings.default_settings.get('lora_4_model', 'None')}"),
                ("", f"{settings.default_settings.get('lora_5_weight', 1.0)} - {settings.default_settings.get('lora_5_model', 'None')}"),
            ],
            'style_selection': settings.default_settings['style'],
            'seed': -1,
            'base_model_name': settings.default_settings['base_model'],
            'performance_selection': settings.default_settings['performance'],
            'aspect_ratios_selection': settings.default_settings["resolution"],
            'cn_selection': None,
            'cn_type': None,

            'image_number': 1,
        }

        # Add work
        task_id = worker.add_task(tmp_data.copy())

        # Wait for result
        finished = False
        while not finished:
            flag, product = worker.task_result(task_id)
            if flag == "results":
                finished = True

        return product

    def api_generate_image(prompt: str) -> str:
        """
        Generates an image from a prompt describing a scene.

        Args:
            prompt (str): Description what the image should look like
        """
        file = _api_process(prompt)[0]

        result = {
            "path": str(file),
            "meta": {"_type": "gradio.FileData"}
        }

        return result
    gr.api(api_generate_image, api_name="generate_image")

    # Search
    def api_search(text: str) -> str:
        """
        Tool to search for images. Uses the filter to find a list of images that match.

        Args:
            filter (str): Text to find images that match.
        Returns:
            images (str): List of image urls
        """
        prompt = f"search: max:10 {text}"
        files = search_pipeline.search(prompt)
        result = []
        for file in files:
            file = Path(file)
            url = shared.local_url + "gradio_api/file=" + re.sub(r'[^/]+/\.\./', '', str(file.relative_to(file.cwd())))
            result.append(url)
        return result
    gr.api(api_search, api_name="search")

    def api_version() -> str:
        """
        Look up the current version of RuinedFooocus

        Returns:
            version (str): The current version of RuinedFooocus
        """
        from version import version
        return version
    gr.api(api_version, api_name="api_version")

    # "secret" pi slideshow
    def get_last_image() -> str:
        """
        Internal use for the secret pi slideshow
        """
        global state
        if "last_image" in state:
            return state["last_image"]
        else:
            return "html/logo.png"
    gr.api(get_last_image, api_name="last_image")
