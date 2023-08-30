import json
from os.path import exists

import modules.path

DEFAULT_SETTINGS = {
    "advanced_mode": False,
    "image_number": 1,
    "seed_random": True,
    "seed": 0,
    "style": "Style: sai-cinematic",
    "prompt": "",
    "negative_prompt": "",
    "performance": "Speed",
    "resolution": "1152x896 (4:3)",
    "sharpness": 2.0,
    "img2img_mode": False,
    "img2img_start_step": 0.06,
    "img2img_denoise": 0.94,
    "base_model": modules.path.default_base_model_name,
    "refiner_model": modules.path.default_refiner_model_name,
    "lora_1_model": modules.path.default_lora_name,
    "lora_1_weight": modules.path.default_lora_weight,
    "lora_2_model": "None",
    "lora_2_weight": modules.path.default_lora_weight,
    "lora_3_model": "None",
    "lora_3_weight": modules.path.default_lora_weight,
    "lora_4_model": "None",
    "lora_4_weight": modules.path.default_lora_weight,
    "lora_5_model": "None",
    "lora_5_weight": modules.path.default_lora_weight,
    "save_metadata": True,
    "theme": "None",
}


def load_settings():
    if exists("settings.json"):
        with open("settings.json") as f:
            settings = json.load(f)
    else:
        # If settings file doesn't exist, create it
        with open("settings.json", "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)
            settings = DEFAULT_SETTINGS

    return settings


default_settings = load_settings()
