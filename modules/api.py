import time
import gradio as gr
from typing import Any
import base64
from pathlib import Path
from shared import (
    state,
    performance_settings,
    resolution_settings,
    path_manager,
)
from modules.settings import default_settings
import modules.search_pipeline as search_pipeline

def add_api():

    # "secret" pi slideshow
    def get_last_image() -> str:
        global state
        if "last_image" in state:
            return state["last_image"]
        else:
            return "html/logo.png"

    gr.api(get_last_image, api_name="last_image")

    # llama
    from modules.llama_pipeline import run_llama
    def api_llama(system: str, user: str) -> str:
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
            'loras': None,

            'style_selection': default_settings['style'],
            'seed': -1,
            'base_model_name': default_settings['base_model'],
            'performance_selection': default_settings['performance'],
            'aspect_ratios_selection': default_settings["resolution"],
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

    def api_prompt2url(prompt: str) -> str:
        file = Path(_api_process(prompt)[0])
        return str(file.relative_to(file.cwd()))

    def api_prompt2img(prompt: str) -> str:
        file = _api_process(prompt)[0]
        with open(file, 'rb') as image:
            image_data = base64.b64encode(image.read())
            result = image_data.decode('ascii')
        return result

    gr.api(api_prompt2url, api_name="prompt2url")
    gr.api(api_prompt2img, api_name="prompt2img")

    # Search
    def api_search(text: str) -> str:
        prompt = f"search: max:10 {text}"
        files = search_pipeline.search(prompt)
        result = []
        for file in files:
            file = Path(file)
            result.append(str(file.relative_to(file.cwd())))
        return result

    gr.api(api_search, api_name="search")