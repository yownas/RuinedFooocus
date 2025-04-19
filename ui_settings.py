import gradio as gr
from shared import path_manager
from pathlib import Path
import json

from modules.settings import default_settings, load_settings


def create_settings():



    with gr.Blocks() as app_settings:
        gr.Markdown("# JSON Settings Editor")

        with gr.Row():
            upload = gr.File(label="Load settings file (optional)", file_count="single", type="filepath")
            download_btn = gr.Button("Download current settings")

        with gr.Row():
            with gr.Column():
                seed_random = gr.Checkbox(label="Seed Random", value=default_settings["seed_random"])
                image_number = gr.Number(label="Image Number", value=default_settings["image_number"])
                image_number_max = gr.Number(label="Image Number Max", value=default_settings["image_number_max"])
                seed = gr.Number(label="Seed", value=default_settings["seed"])
                style = gr.Textbox(label="Style", value=default_settings["style"])
                prompt = gr.Textbox(label="Prompt", value=default_settings["prompt"])
                negative_prompt = gr.Textbox(label="Negative Prompt", value=default_settings.get("negative_prompt", ""))
                performance = gr.Textbox(label="Performance", value=default_settings["performance"])
                resolution = gr.Textbox(label="Resolution", value="Should be a Copy From UI button")
                base_model = gr.Textbox(label="Base Model", value=default_settings["performance"])
                lora_1_model = gr.Textbox(label="Lora 1 Model")
                lora_1_weight = gr.Number(label="Lora 1 Weight", value=1.0)
                lora_2_model = gr.Textbox(label="Lora 2 Model")
                lora_2_weight = gr.Number(label="Lora 2 Weight", value=0.5)
                lora_3_model = gr.Textbox(label="Lora 3 Model")
                lora_3_weight = gr.Number(label="Lora 3 Weight", value=0.5)
                lora_4_model = gr.Textbox(label="Lora 4 Model")
                lora_4_weight = gr.Number(label="Lora 4 Weight", value=0.5)
                lora_5_model = gr.Textbox(label="Lora 5 Model")
                lora_5_weight = gr.Number(label="Lora 5 Weight", value=0.5)
                theme = gr.Textbox(label="Theme", value=default_settings.get("theme", ""))
                save_metadata = gr.Checkbox(label="Save Metadata", value=default_settings.get("save_metadata", True))
                auto_negative_prompt = gr.Checkbox(label="Auto Negative Prompt", value=default_settings["auto_negative_prompt"])
                OBP_preset = gr.Textbox(label="OBP Preset", value=default_settings["OBP_preset"])
                hint_chance = gr.Number(label="Hint Chance", value=default_settings.get("hint_chance", 25))
                gguf_clip1 = gr.Textbox(label="gguf_clip1", value=default_settings.get("gguf_clip1", ""))
                gguf_clip2 = gr.Textbox(label="gguf_clip2", value=default_settings.get("gguf_clip2", ""))
                clip_g = gr.Textbox(label="clip_g", value=default_settings.get("clip_g", ""))
                clip_l = gr.Textbox(label="clip_l", value=default_settings.get("clip_l", ""))
                clip_t5 = gr.Textbox(label="clip_t5", value=default_settings.get("clip_t5", ""))
                llama_localfile = gr.Textbox(label="llama_localfile", value=default_settings.get("llama_localfile", ""))

        with gr.Row():
            save_btn = gr.Button("Save Settings")
            output = gr.Textbox(label="Status")
            download_file = gr.File(label="Download File")


    return app_settings


