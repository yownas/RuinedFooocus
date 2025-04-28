import json
from os.path import exists
from pathlib import Path
import time
import shared

class SettingsManager:
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
    settings_path = None
    name = None

    def __init__(self):
        from argparser import args
        self.name = args.settings
        self.set_settings_path(args.settings)
        self.load_settings()

    def set_settings_path(self, subfolder=None):
        self.subfolder = subfolder
        if self.subfolder in [None, "", "default"]:
            path = Path("settings/settings.json")
        else:
            path = Path(f"settings/{self.subfolder}/settings.json")
        if not path.parent.exists():
            path.parent.mkdir()
        self.settings_path = path

    def load_settings(self):
        path = self.settings_path
        if path and exists(path):
            with open(path) as f:
                self.default_settings = json.load(f)
        else:
            self.default_settings = {}

        # Add any missing default settings
        changed = False
        for key, value in self.DEFAULT_SETTINGS.items():
            if key not in self.default_settings:
                self.default_settings[key] = value
                changed = True

        # Some sanity checks
        for key in ['style', 'archive_folders']:
            if key in self.default_settings and not isinstance(self.default_settings[key], list):
                self.default_settings[key] = [self.default_settings[key]]

        if changed:
            with open(self.settings_path, "w") as f:
                json.dump(self.default_settings, f, indent=2)

    def save_settings(self):
        # FIXME: Add some error checks and exception handling
        with open(self.settings_path, "w") as f:
            json.dump(self.default_settings, f, indent=2)
        shared.update_cfg()
