from pathlib import Path
import json

from modules.civit import Civit


class PathManager:
    DEFAULT_PATHS = {
        "path_checkpoints": "../models/checkpoints/",
        "path_loras": "../models/loras/",
        "path_controlnet": "../models/controlnet/",
        "path_vae_approx": "../models/vae_approx/",
        "path_preview": "../outputs/preview.jpg",
        "path_upscalers": "../models/upscale_models",
        "path_outputs": "../outputs/",
        "path_clip": "../models/clip/",
    }

    EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors"]

    def __init__(self):
        self.paths = self.load_paths()
        self.model_paths = self.get_model_paths()
        self.default_model_names = self.get_default_model_names()
        self.update_all_model_names()

    def load_paths(self):
        paths = self.DEFAULT_PATHS.copy()
        settings_path = Path("settings/paths.json")
        if settings_path.exists():
            with settings_path.open() as f:
                paths.update(json.load(f))
        for key in self.DEFAULT_PATHS:
            if key not in paths:
                paths[key] = self.DEFAULT_PATHS[key]
        with settings_path.open("w") as f:
            json.dump(paths, f, indent=2)
        return paths

    def get_model_paths(self):
        return {
            "modelfile_path": self.get_abspath_folder(self.paths["path_checkpoints"]),
            "lorafile_path": self.get_abspath_folder(self.paths["path_loras"]),
            "controlnet_path": self.get_abspath_folder(self.paths["path_controlnet"]),
            "vae_approx_path": self.get_abspath_folder(self.paths["path_vae_approx"]),
            "temp_outputs_path": self.get_abspath_folder(self.paths["path_outputs"]),
            "temp_preview_path": self.get_abspath(self.paths["path_preview"]),
            "upscaler_path": self.get_abspath_folder(self.paths["path_upscalers"]),
            "clip_path": self.get_abspath_folder(self.paths["path_clip"]),
        }

    def get_default_model_names(self):
        return {
            "default_base_model_name": "sd_xl_base_1.0_0.9vae.safetensors",
            "default_lora_name": "sd_xl_offset_example-lora_1.0.safetensors",
            "default_lora_weight": 0.5,
        }

    def get_abspath_folder(self, path):
        folder = self.get_abspath(path)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_abspath(self, path):
        return Path(path) if Path(path).is_absolute() else Path(__file__).parent / path

    def get_model_filenames(self, folder_path, isLora=False):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError("Folder path is not a valid directory.")
        filenames = []
        civit = Civit()
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                if isLora:
                    txtcheck = path.with_suffix(".txt")
                    if txtcheck.exists():
                        path = path.with_suffix(f"{path.suffix} 🗒️")
                    else:
                        hash = civit.model_hash(str(path))
                        models = civit.get_models_by_hash(hash)
                        keywords = civit.get_keywords(models)
                        with open(txtcheck, "w") as f:
                            f.write(" ".join(keywords))
                filenames.append(str(path.relative_to(folder_path)))
        return sorted(
            filenames,
            key=lambda x: f"0{x.casefold()}"
            if Path(x).suffix in x
            else f"1{x.casefold()}",
        )

    def update_all_model_names(self):
        self.model_filenames = self.get_model_filenames(
            self.model_paths["modelfile_path"]
        )
        self.lora_filenames = self.get_model_filenames(
            self.model_paths["lorafile_path"], True
        )
        self.upscaler_filenames = self.get_model_filenames(
            self.model_paths["upscaler_path"]
        )

    def find_lcm_lora(self):
        path = Path(self.model_paths["lorafile_path"])
        filename = "lcm-lora-sdxl.safetensors"
        for child in path.rglob(filename):
            if child.name == filename:
                return child.relative_to(path)
