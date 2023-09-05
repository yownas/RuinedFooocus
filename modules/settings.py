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
        settings = {}

    # Add any missing default settings
    changed = False
    for key, value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = value
            changed = True

    if changed:
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)

    return settings


default_settings = load_settings()
