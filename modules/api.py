import gradio as gr
from shared import (
    state,
    performance_settings,
    resolution_settings,
    path_manager,
)

def add_api():

    def get_last_image() -> str:
        global state
        if "last_image" in state:
            return state["last_image"]
        else:
            return "html/logo.png"

    gr.api(get_last_image, api_name="last_image")

