import gradio as gr
from shared import path_manager
import modules.async_worker as worker
from pathlib import Path
import json


def create_settings():
    with gr.Blocks() as app_settings:
        with gr.Row():
            placeholder = gr.Image(
                label="Placeholder",
                value="html/error.png"
            )

    return app_settings


