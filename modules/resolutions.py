import json
import shutil
from pathlib import Path


class ResolutionSettings:
    DEFAULT_RESOLUTIONS_FILE = Path("settings/resolutions.default")
    RESOLUTIONS_FILE = Path("settings/resolutions.json")
    CUSTOM_RESOLUTION = "Custom..."

    def __init__(self):
        self.load_resolutions()

    def load_resolutions(self):
        self.base_ratios = {}

        if not self.RESOLUTIONS_FILE.is_file():
            shutil.copy(self.DEFAULT_RESOLUTIONS_FILE, self.RESOLUTIONS_FILE)

        with self.RESOLUTIONS_FILE.open() as f:
            data = json.load(f)
            for ratio, res in data.items():
                self.base_ratios[ratio] = (res["width"], res["height"])

        self.aspect_ratios = {f"{v[0]}x{v[1]} ({k})": v for k, v in self.base_ratios.items()}

        return self.base_ratios

    def save_resolutions(self, res_options):
        formatted_options = {}
        for k in res_options:
            formatted_options[k] = {
                "width":  res_options[k][0],
                "height": res_options[k][1],
            }

        with open(self.RESOLUTIONS_FILE, "w") as f:
            json.dump(formatted_options, f, indent=2)

        return self.load_resolutions()

    def get_base_aspect_ratios(self, name):
        return self.base_ratios[name]

    def get_aspect_ratios(self, name):
        return self.aspect_ratios[name]
