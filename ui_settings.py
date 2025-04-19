import gradio as gr
from shared import path_manager
from pathlib import Path
import json

from modules.settings import default_settings, load_settings


def create_settings():



    with gr.Blocks() as app_settings:
#        gr.Markdown("# JSON Settings Editor")

        with gr.Row():
            with gr.Column():
                gr.Markdown("# UI settings")
                seed_random = gr.Checkbox(label="Seed Random", value=default_settings.get("seed_random", True))
                image_number = gr.Number(label="Image Number", value=default_settings.get("image_number", 1))
                image_number_max = gr.Number(label="Image Number Max", value=default_settings.get("image_number_max", 50))
                seed = gr.Number(label="Seed", value=default_settings.get("seed", -1))
                style = gr.Textbox(label="Style", value=default_settings.get("style", "[]"))
                prompt = gr.Textbox(label="Prompt", value=default_settings.get("prompt", ""))
                negative_prompt = gr.Textbox(label="Negative Prompt", value=default_settings.get("negative_prompt", ""))
                performance = gr.Textbox(label="Performance", value=default_settings.get("performance", "Speed"))
                resolution = gr.Textbox(label="Resolution", value="Should be a Copy From UI button")
                base_model = gr.Textbox(label="Base Model", value=default_settings.get("base_model", ""))
#                lora_1_model = gr.Textbox(label="Lora 1 Model")
#                lora_1_weight = gr.Number(label="Lora 1 Weight", value=1.0)
#                lora_2_model = gr.Textbox(label="Lora 2 Model")
#                lora_2_weight = gr.Number(label="Lora 2 Weight", value=0.5)
#                lora_3_model = gr.Textbox(label="Lora 3 Model")
#                lora_3_weight = gr.Number(label="Lora 3 Weight", value=0.5)
#                lora_4_model = gr.Textbox(label="Lora 4 Model")
#                lora_4_weight = gr.Number(label="Lora 4 Weight", value=0.5)
#                lora_5_model = gr.Textbox(label="Lora 5 Model")
#                lora_5_weight = gr.Number(label="Lora 5 Weight", value=0.5)
                save_metadata = gr.Checkbox(label="Save Metadata", value=default_settings.get("save_metadata", True))
                auto_negative_prompt = gr.Checkbox(label="Auto Negative Prompt", value=default_settings.get("auto_negative_prompt", False))
            with gr.Column():
                gr.Markdown("# One Button Prompt")
                OBP_preset = gr.Textbox(label="OBP Preset", value=default_settings.get("OBP_preset", "Standard"))
                hint_chance = gr.Number(label="Hint Chance", value=default_settings.get("hint_chance", 25))
            with gr.Column():
                gr.Markdown("# Other")
                theme = gr.Textbox(label="Theme", value=default_settings.get("theme", ""))
                clip_g = gr.Textbox(label="clip_g", value=default_settings.get("clip_g", ""))
                clip_l = gr.Textbox(label="clip_l", value=default_settings.get("clip_l", ""))
                clip_t5 = gr.Textbox(label="clip_t5", value=default_settings.get("clip_t5", ""))
                llama_localfile = gr.Textbox(label="llama_localfile", value=default_settings.get("llama_localfile", ""))

        with gr.Row():
            save_btn = gr.Button("Save Settings")
            output = gr.Textbox(label="Status")
            download_file = gr.File(label="Download File")
        with gr.Row():
            upload = gr.File(label="Load settings file (optional)", file_count="single", type="filepath")
            download_btn = gr.Button("Download current settings")



    return app_settings


