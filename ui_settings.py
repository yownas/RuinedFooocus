import gradio as gr
from pathlib import Path
import json

from modules.settings import default_settings, load_settings, save_settings

from shared import state, add_setting

def save_clicked(*args):
    global default_settings

    # Overwrite current settings
    for key, val in zip(state["setting_name"], args):
        default_settings[key] = val

        # Remove empty keys
        if default_settings[key] == None or default_settings[key] == "":
            default_settings.pop(key)

    save_settings()
    gr.Info("Saved!")


def create_settings():
    with gr.Blocks() as app_settings:
#        gr.Markdown("# JSON Settings Editor")
        with gr.Row():
            with gr.Column():
                gr.Markdown("# UI settings")
                image_number = gr.Number(label="Image Number", interactive=True, value=default_settings.get("image_number", 1))
                image_number_max = gr.Number(label="Image Number Max", interactive=True, value=default_settings.get("image_number_max", 50))
                seed_random = gr.Checkbox(label="Seed Random", interactive=True, value=default_settings.get("seed_random", True))
                seed = gr.Number(label="Seed", interactive=True, value=default_settings.get("seed", -1))
                style = gr.Textbox(label="Style", interactive=True, value=default_settings.get("style", "[]"))
                prompt = gr.Textbox(label="Prompt", interactive=True, value=default_settings.get("prompt", ""))
                negative_prompt = gr.Textbox(label="Negative Prompt", interactive=True, value=default_settings.get("negative_prompt", ""))
                performance = gr.Textbox(label="Performance", interactive=True, value=default_settings.get("performance", "Speed"))
                resolution = gr.Textbox(label="Resolution", interactive=True, value="Should be a Copy From UI button")
                base_model = gr.Textbox(label="Base Model", interactive=True, value=default_settings.get("base_model", ""))
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
                auto_negative_prompt = gr.Checkbox(label="Auto Negative Prompt", interactive=True, value=default_settings.get("auto_negative_prompt", False))
            with gr.Column():
                gr.Markdown("# One Button Prompt")
                OBP_preset = gr.Textbox(label="OBP Preset", value=default_settings.get("OBP_preset", "Standard"))
                hint_chance = gr.Number(label="Hint Chance", value=default_settings.get("hint_chance", 25))
                
                gr.Markdown("# Image Browser")
                images_per_page = gr.Textbox(label="Images per page", value=default_settings.get("images_per_page", 100))
                archive_folders = gr.Code(
                    label="Archive folders",
                    interactive=True,
                    value="\n".join(default_settings.get("archive_folders", [])),
                    lines=5,
                    max_lines=5
                )
                gr.Markdown("# Paths")

            with gr.Column():
                gr.Markdown("# Other")
                interrogator = gr.Textbox(label="Default interrogator", interactive=True, placeholder="florence", value=default_settings.get("interrogator", None))
                add_setting("interrogator", interrogator)
                save_metadata = gr.Checkbox(label="Save Metadata", value=default_settings.get("save_metadata", True))
                add_setting("save_metadata", save_metadata)
                theme = gr.Textbox(label="Theme", interactive=True, value=default_settings.get("theme", None))
                add_setting("theme", theme)
                clip_g = gr.Textbox(label="clip_g", interactive=True, placeholder="clip_g.safetensors", value=default_settings.get("clip_g", None))
                add_setting("clip_g", clip_g)
                clip_t = gr.Textbox(label="clip_t", interactive=True, placeholder="clip_t.safetensors", value=default_settings.get("clip_t", None))
                add_setting("clip_t", clip_t)
                clip_t5 = gr.Textbox(label="clip_t5", interactive=True, placeholder="clip_t5.safetensors", value=default_settings.get("clip_t5", None))
                add_setting("clip_t5", clip_t5)
                clip_umt5 = gr.Textbox(label="clip_umt5", interactive=True, placeholder="clip_umt5.safetensors", value=default_settings.get("clip_umt5", None))
                add_setting("clip_umt5", clip_umt5)
                clip_llava = gr.Textbox(label="clip_llava", interactive=True, placeholder="clip_llava.safetensors", value=default_settings.get("clip_llava", None))
                add_setting("clip_llava", clip_llava)
                clip_vision = gr.Textbox(label="clip_vision", interactive=True, placeholder="clip_vision.safetensors", value=default_settings.get("clip_vision", None))
                add_setting("clip_vision", clip_vision)

                lumina2_shift = gr.Textbox(label="Lumina2 shift", interactive=True, placeholder=3.0, value=default_settings.get("lumina2_shift", None))
                add_setting("lumina2_shift", lumina2_shift)

                vae_flux = gr.Textbox(label="Flux VAE", interactive=True, placeholder="ae.safetensors", value=default_settings.get("vae_flux", None))
                add_setting("vae_flux", vae_flux)
                vae_hunyuan_video = gr.Textbox(label="Hunyuan Video VAE", interactive=True, placeholder="hunyuan_video_vae_bf16.safetensors", value=default_settings.get("vae_hunyuan_video", None))
                add_setting("vae_hunyuan_video", vae_hunyuan_video)
                vae_lumina2 = gr.Textbox(label="Lumina2 VAE", interactive=True, placeholder="lumina2_vae_fp32.safetensors", value=default_settings.get("vae_lumina2", None))
                add_setting("vae_lumina2", vae_lumina2)
                vae_sd3 = gr.Textbox(label="SD3 VAE", interactive=True, placeholder="sd3_vae.safetensors", value=default_settings.get("vae_sd3", None))
                add_setting("vae_sd3", vae_sd3)
                vae_sdxl = gr.Textbox(label="SDXL VAE", interactive=True, placeholder="sdxl_vae.safetensors", value=default_settings.get("vae_sdxl", None))
                add_setting("vae_sdxl", vae_sdxl)
                vae_wan = gr.Textbox(label="WAN 2.1 VAE", interactive=True, placeholder="wan_2.1_vae.safetensors", value=default_settings.get("vae_wan", None))
                add_setting("vae_wan", vae_wan)

                llama_localfile = gr.Textbox(label="llama_localfile", interactive=True, placeholder="", value=default_settings.get("llama_localfile", None))
                add_setting("llama_localfile", llama_localfile)

#modules/llama_pipeline.py:        repo = default_settings.get("llama_repo", "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF")
#modules/llama_pipeline.py:        file = default_settings.get("llama_file", "*q8_0.gguf")



        with gr.Row():
            save_btn = gr.Button("Save Settings")

# Deal with this later
#            output = gr.Textbox(label="Status")
#            download_file = gr.File(label="Download File")
#        with gr.Row():
#            upload = gr.File(label="Load settings file (optional)", file_count="single", type="filepath")
#            download_btn = gr.Button("Download current settings")


        save_btn.click(
            fn=save_clicked,
            inputs=state["setting_obj"],
            outputs=[
            ]
        )

    return app_settings


