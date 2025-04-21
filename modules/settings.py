import json
from os.path import exists
from pathlib import Path

class SettingsManager():
    DEFAULT_SETTINGS = {
        "advanced_mode": False,
        "image_number": 1,
        "seed_random": True,
        "seed": 0,
        "style": ["Style: sai-cinematic"],
        "prompt": "",
        "negative_prompt": "",
        "performance": "Speed",
        "resolution": "1152x896 (4:3)",
        "base_model": "sd_xl_base_1.0_0.9vae.safetensors",
        "lora_1_model": "None",
        "lora_1_weight": 0.5,
        "lora_2_model": "None",
        "lora_2_weight": 0.5,
        "lora_3_model": "None",
        "lora_3_weight": 0.5,
        "lora_4_model": "None",
        "lora_4_weight": 0.5,
        "lora_5_model": "None",
        "lora_5_weight": 0.5,
        "theme": "None",
        "auto_negative_prompt": False,
        "OBP_preset": "Standard",
        "hint_chance": 25,
    }

    default_settings = None

    def __init__(self):
        self.settings_path = Path("settings/settings.json")
        self.default_settings = self.load_settings()

    def load_settings(self):
        if exists(self.settings_path):
            with open(self.settings_path) as f:
                settings = json.load(f)
        else:
            settings = {}

        # Add any missing default settings
        changed = False
        for key, value in self.DEFAULT_SETTINGS.items():
            if key not in settings:
                settings[key] = value
                changed = True

        # Some sanity checks
        if not isinstance(settings["style"], list):
            settings["style"] = []

        if changed:
            with open(self.settings_path, "w") as f:
                json.dump(settings, f, indent=2)

        return settings

    def save_settings(self):
        # FIXME: Add some error checks and exception handling
        with open(self.settings_path, "w") as f:
            json.dump(self.default_settings, f, indent=2)
