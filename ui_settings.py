import gradio as gr
from pathlib import Path

from modules.interrogate import looks

from shared import state, add_setting, performance_settings, resolution_settings, path_manager, settings, models

def save_clicked(*args):
    ui_data = {}
    # Overwrite current settings
    for key, val in zip(state["setting_name"], args):
        settings.default_settings[key] = val

        # Massage some of the data. Settings that are lists should be split
        if key in [
            "archive_folders",
            "style",
            "path_checkpoints",
            "path_loras"
        ]:
            settings.default_settings[key] = settings.default_settings[key].splitlines()

        # Remove empty keys
        if settings.default_settings[key] == None or settings.default_settings[key] == "":
            settings.default_settings.pop(key)
            continue

        # Move ui_* and path_*
        if key.startswith("ui_"):
            ui_data[key] = settings.default_settings.get(key)
            settings.default_settings.pop(key)
        if key.startswith("path_"):
            path_manager.paths[key] = settings.default_settings.get(key)
            settings.default_settings.pop(key)

    settings.set_settings_path(ui_data.get("ui_settings_name", None))
    settings.save_settings()
    path_manager.set_settings_path(ui_data.get("ui_settings_name", None))
    path_manager.save_paths()

    print(f"Saved new settings to {settings.settings_path}. Please restart.") 
    gr.Info("Saved! Please restart RuinedFoocus.")


def create_settings():
    with gr.Blocks() as app_settings:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# UI settings")
                image_number = gr.Number(label="Image Number", interactive=True, value=settings.default_settings.get("image_number", 1))
                add_setting("image_number", image_number)
                image_number_max = gr.Number(label="Image Number Max", interactive=True, value=settings.default_settings.get("image_number_max", 50))
                add_setting("image_number_max", image_number_max)
                seed_random = gr.Checkbox(label="Seed Random", interactive=True, value=settings.default_settings.get("seed_random", True))
                add_setting("seed_random", seed_random)
                seed = gr.Number(label="Seed", interactive=True, value=settings.default_settings.get("seed", -1))
                add_setting("seed", seed)
                style = gr.Code(
                    label="Style",
                    interactive=True,
                    value="\n".join(settings.default_settings.get("style", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("style", style)
                prompt = gr.Textbox(label="Prompt", interactive=True, value=settings.default_settings.get("prompt", ""))
                add_setting("prompt", prompt)
                negative_prompt = gr.Textbox(label="Negative Prompt", interactive=True, value=settings.default_settings.get("negative_prompt", ""))
                add_setting("negative_prompt", negative_prompt)
                performance = gr.Dropdown(
                    label="Performance",
                    interactive=True,
                    choices=list(performance_settings.performance_options.keys()),
                    value=settings.default_settings.get("performance", "Speed"),
                )
                add_setting("performance", performance)
                resolution = gr.Dropdown(
                    label="Resolution",
                    interactive=True,
                    choices=list(resolution_settings.aspect_ratios.keys()),
                    value=settings.default_settings.get("resolution", "1344x768 (16:9)"),
                )
                add_setting("resolution", resolution)
                base_model = gr.Dropdown(
                    label="Base Model",
                    interactive=True,
                    choices=models.names['checkpoints'],
                    value=settings.default_settings.get("base_model", "sd_xl_base_1.0_0.9vae.safetensors"),
                )
                add_setting("base_model", base_model)

                with gr.Row():
                    lora_1_model = gr.Dropdown(
                        label="LoRA 1 Model",
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_1_model", "None"),
                    )
                    lora_1_weight = gr.Number(label="Lora 1 Weight", value=settings.default_settings.get("lora_1_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_2_model = gr.Dropdown(
                        label="LoRA 2 Model",
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_2_model", "None"),
                    )
                    lora_2_weight = gr.Number(label="Lora 2 Weight", value=settings.default_settings.get("lora_2_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_3_model = gr.Dropdown(
                        label="LoRA 3 Model",
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_3_model", "None"),
                    )
                    lora_3_weight = gr.Number(label="Lora 3 Weight", value=settings.default_settings.get("lora_3_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_4_model = gr.Dropdown(
                        label="LoRA 4 Model",
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_4_model", "None"),
                    )
                    lora_4_weight = gr.Number(label="Lora 4 Weight", value=settings.default_settings.get("lora_4_weight", 1.0), step=0.05)
                with gr.Row():
                    lora_5_model = gr.Dropdown(
                        label="LoRA 5 Model",
                        interactive=True,
                        choices=["None"] + models.names['loras'],
                        value=settings.default_settings.get("lora_5_model", "None"),
                    )
                    lora_5_weight = gr.Number(label="Lora 5 Weight", value=settings.default_settings.get("lora_5_weight", 1.0), step=0.05)

                add_setting("lora_1_model", lora_1_model)
                add_setting("lora_2_model", lora_2_model)
                add_setting("lora_3_model", lora_3_model)
                add_setting("lora_4_model", lora_4_model)
                add_setting("lora_5_model", lora_5_model)
                add_setting("lora_1_weight", lora_1_weight)
                add_setting("lora_2_weight", lora_2_weight)
                add_setting("lora_3_weight", lora_3_weight)
                add_setting("lora_4_weight", lora_4_weight)
                add_setting("lora_5_weight", lora_5_weight)

                auto_negative_prompt = gr.Checkbox(label="Auto Negative Prompt", interactive=True, value=settings.default_settings.get("auto_negative_prompt", False))
                add_setting("auto_negative_prompt", auto_negative_prompt)
            with gr.Column():
                gr.Markdown("# One Button Prompt")
                OBP_preset = gr.Textbox(label="OBP Preset", value=settings.default_settings.get("OBP_preset", "Standard"))
                add_setting("OBP_preset", OBP_preset)
                hint_chance = gr.Number(label="Hint Chance", value=settings.default_settings.get("hint_chance", 25))
                add_setting("hint_chance", hint_chance)
                
                gr.Markdown("# Image Browser")
                images_per_page = gr.Number(label="Images per page", value=settings.default_settings.get("images_per_page", 100), minimum=1, maximum=1000, step=1)
                add_setting("images_per_page", images_per_page)
                archive_folders = gr.Code(
                    label="Archive folders",
                    interactive=True,
                    value="\n".join(settings.default_settings.get("archive_folders", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("archive_folders", archive_folders)
                gr.Markdown("# Paths")
                path_checkpoints = gr.Code(
                    label="Checkpoint folders",
                    interactive=True,
                    value="\n".join(path_manager.paths.get("path_checkpoints", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("path_checkpoints", path_checkpoints)
                path_loras = gr.Code(
                    label="LoRA folders",
                    interactive=True,
                    value="\n".join(path_manager.paths.get("path_loras", [])),
                    lines=5,
                    max_lines=5
                )
                add_setting("path_loras", path_loras)
                path_inbox = gr.Textbox(label="Inbox folder", interactive=True, placeholder="", value=path_manager.paths.get("path_inbox", "../models/inbox/"))
                add_setting("path_inbox", path_inbox)
                path_outputs = gr.Textbox(label="Output folder", interactive=True, placeholder="", value=path_manager.paths.get("path_outputs", "../outputs/"))
                add_setting("path_outputs", path_outputs)

            with gr.Column():
                gr.Markdown("# Other")
                interrogator = gr.Dropdown(label="Default interrogator", interactive=True, choices=list(looks.keys()), value=settings.default_settings.get("interrogator", None),)
                add_setting("interrogator", interrogator)
                save_metadata = gr.Checkbox(label="Save Metadata", value=settings.default_settings.get("save_metadata", True))
                add_setting("save_metadata", save_metadata)
                theme = gr.Textbox(label="Theme", interactive=True, value=settings.default_settings.get("theme", None))
                add_setting("theme", theme)
                clip_g = gr.Dropdown(label="clip_g", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_g", None),)
                add_setting("clip_g", clip_g)
                clip_gemma = gr.Dropdown(label="clip_gemma", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_gemma", None),)
                add_setting("clip_gemma", clip_gemma)
                clip_l = gr.Dropdown(label="clip_l", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_l", None),)
                add_setting("clip_l", clip_l)
                clip_llava = gr.Dropdown(label="clip_llava", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_llava", None),)
                add_setting("clip_llava", clip_llava)
                clip_t5 = gr.Dropdown(label="clip_t5", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_t5", None),)
                add_setting("clip_t5", clip_t5)
                clip_umt5 = gr.Dropdown(label="clip_umt5", interactive=True, choices=[None]+path_manager.get_folder_list("clip"), value=settings.default_settings.get("clip_umt5", None),)
                add_setting("clip_umt5", clip_umt5)
                clip_vision = gr.Dropdown(label="clip_vision", interactive=True, choices=[None]+path_manager.get_folder_list("clip_vision"), value=settings.default_settings.get("clip_vision", None),)
                add_setting("clip_vision", clip_vision)

                lumina2_shift = gr.Textbox(label="Lumina2 shift", interactive=True, placeholder=3.0, value=settings.default_settings.get("lumina2_shift", None))
                add_setting("lumina2_shift", lumina2_shift)

                vae_flux = gr.Dropdown(label="Flux VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_flux", None),)
                add_setting("vae_flux", vae_flux)
                vae_hunyuan_video = gr.Dropdown(label="Hunyuan Video VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_hunyuan_video", None),)
                add_setting("vae_hunyuan_video", vae_hunyuan_video)
                vae_lumina2 = gr.Dropdown(label="Lumina2 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_lumina2", None),)
                add_setting("vae_lumina2", vae_lumina2)
                vae_sd3 = gr.Dropdown(label="SD3 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sd3", None),)
                add_setting("vae_sd3", vae_sd3)
                vae_sdxl = gr.Dropdown(label="SDXL/Pony/Illustrious VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_sdxl", None),)
                add_setting("vae_sdxl", vae_sdxl)
                vae_wan = gr.Dropdown(label="WAN 2.1 VAE", interactive=True, choices=[None]+path_manager.get_folder_list("vae"), value=settings.default_settings.get("vae_wan", None),)
                add_setting("vae_wan", vae_wan)
                llama_localfile = gr.Dropdown(label="Local Llama file", interactive=True, choices=[None]+path_manager.get_folder_list("llm"), value=settings.default_settings.get("llama_localfile", None),)
                add_setting("llama_localfile", llama_localfile)

        with gr.Row(), gr.Group():
            ui_settings_name = gr.Text(
                label="Settings name",
                interactive=True,
                placeholder="Optional",
                value=settings.name,
            )
            add_setting("ui_settings_name", ui_settings_name)
            save_btn = gr.Button("Save Settings")

# Deal with this later
#            output = gr.Textbox(label="Status")
#            download_file = gr.File(label="Download File")
#        with gr.Row():
#            upload = gr.File(label="Load settings file (optional)", file_count="single", type="filepath")
#            download_btn = gr.Button("Download current settings")


        save_btn.click(
            fn=save_clicked,
            show_api=False,
            inputs=state["setting_obj"],
        )

    return app_settings


