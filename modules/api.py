import time
import gradio as gr
from typing import Any
import base64
from shared import (
    state,
    performance_settings,
    resolution_settings,
    path_manager,
)
from modules.settings import default_settings

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
    def api_process(prompt: str) -> str:
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

        # TODO: Wait until queue is empty?
        # Add work
        worker.buffer.append({"task_type": "start", "image_total": 1})
        worker.buffer.append(tmp_data.copy())

        # Wait for result
        finished = False
        while not finished:
            time.sleep(0.1)

            if not worker.outputs:
                continue

            flag, product = worker.outputs.pop(0)

# Ingore previews
#            if flag == "preview":
#                yield update_preview(product)

            if flag == "results":
                finished = True

        worker.buffer.append({"task_type": "stop"})

        with open(product[0], 'rb') as image:
            image_data = base64.b64encode(image.read())
            results = image_data.decode('ascii')

        return results

    gr.api(api_process, api_name="process")
