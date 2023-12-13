import json
import shutil
from pathlib import Path


class ResolutionSettings:
    DEFAULT_RESOLUTIONS_FILE = Path("settings/resolutions.default")
    RESOLUTIONS_FILE = Path("settings/resolutions.json")
    CUSTOM_RESOLUTION = "Custom..."

    def __init__(self):
        base_rations = self.load_resolutions()
        self.aspect_ratios = {f"{v[0]}x{v[1]} ({k})": v for k, v in base_rations.items()}

    def load_resolutions(self):
        ratios = {}

        if not self.RESOLUTIONS_FILE.is_file():
            shutil.copy(self.DEFAULT_RESOLUTIONS_FILE, self.RESOLUTIONS_FILE)

        with self.RESOLUTIONS_FILE.open() as f:
            data = json.load(f)
            for ratio, res in data.items():
                ratios[ratio] = (res["width"], res["height"])

        return ratios


    def get_aspect_ratios(self):
        return self.aspect_ratios
