from pathlib import Path
import json
import os
import requests
from tqdm import tqdm


class PathManager:
    DEFAULT_PATHS = {
        "path_checkpoints": "../models/checkpoints/",
        "path_diffusers": "../models/diffusers/",
        "path_diffusers_cache": "../models/diffusers_cache/",
        "path_loras": "../models/loras/",
        "path_controlnet": "../models/controlnet/",
        "path_vae_approx": "../models/vae_approx/",
        "path_vae": "../models/vae/",
        "path_preview": "../outputs/preview.jpg",
        "path_faceswap": "../models/faceswap/",
        "path_upscalers": "../models/upscale_models",
        "path_outputs": "../outputs/",
        "path_clip": "../models/clip/",
        "path_cache": "../cache/",
        "path_llm": "../models/llm",
    }

    EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors", ".gguf", ".merge"]

    # Add a dictionary to store file download information
    DOWNLOADABLE_FILES = {}

    def __init__(self):
        self.paths = self.load_paths()
        self.model_paths = self.get_model_paths()
        self.default_model_names = self.get_default_model_names()
        self.update_all_model_names()

        pathdb_folder = "modules/pathdb"
        files = os.listdir(pathdb_folder)
        for file in files:
            # Check if the file has a .json extension
            if file.endswith('.json'):
                file_path = os.path.join(pathdb_folder, file)

                try:
                    # Open and read the JSON file
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        self.DOWNLOADABLE_FILES.update(data)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

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
            "diffusers_path": self.get_abspath_folder(self.paths["path_diffusers"]),
            "diffusers_cache_path": self.get_abspath_folder(
                self.paths["path_diffusers_cache"]
            ),
            "lorafile_path": self.get_abspath_folder(self.paths["path_loras"]),
            "controlnet_path": self.get_abspath_folder(self.paths["path_controlnet"]),
            "vae_approx_path": self.get_abspath_folder(self.paths["path_vae_approx"]),
            "vae_path": self.get_abspath_folder(self.paths["path_vae"]),
            "temp_outputs_path": self.get_abspath_folder(self.paths["path_outputs"]),
            "temp_preview_path": self.get_abspath(self.paths["path_preview"]),
            "faceswap_path": self.get_abspath_folder(self.paths["path_faceswap"]),
            "upscaler_path": self.get_abspath_folder(self.paths["path_upscalers"]),
            "clip_path": self.get_abspath_folder(self.paths["path_clip"]),
            "cache_path": self.get_abspath_folder(self.paths["path_cache"]),
            "llm_path": self.get_abspath_folder(self.paths["path_llm"]),
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

    def get_model_filenames(self, folder_path, cache=None, isLora=False):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")
        filenames = []
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:
                if isLora:
                    txtcheck = path.with_suffix(".txt")
                    if txtcheck.exists():
                        fstats = txtcheck.stat()
                        if fstats.st_size > 0:
                            path = path.with_suffix(f"{path.suffix}")
                filenames.append(str(path.relative_to(folder_path)))
        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        return sorted(
            filenames,
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )

    def get_diffusers_filenames(self, folder_path, cache=None, isLora=False):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")
        filenames = []
        for path in folder_path.glob("*/*"):
            #            if path.suffix.lower() in self.EXTENSIONS:
            #                if isLora:
            #                    txtcheck = path.with_suffix(".txt")
            #                    if txtcheck.exists():
            #                        fstats = txtcheck.stat()
            #                        if fstats.st_size > 0:
            #                            path = path.with_suffix(f"{path.suffix}")
            filenames.append(f"🤗:{path.relative_to(folder_path)}")
        return sorted(
            filenames,
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )

    def update_all_model_names(self):
        self.model_filenames = self.get_model_filenames(
            self.model_paths["modelfile_path"], cache="checkpoints"
        ) + self.get_diffusers_filenames(
            self.model_paths["diffusers_path"], cache="checkpoints"
        )
        self.lora_filenames = self.get_model_filenames(
            self.model_paths["lorafile_path"], cache="loras", isLora=True
        )
        self.upscaler_filenames = self.get_model_filenames(
            self.model_paths["upscaler_path"]
        )

    def get_file_path(self, file_key, default=None):
        """
        Get the path for a file, downloading it if it doesn't exist.
        """
        if file_key not in self.DOWNLOADABLE_FILES:
#            if default is None:
#                raise ValueError(f"Unknown file key: {file_key}")
#           else:
            return default

        file_info = self.DOWNLOADABLE_FILES[file_key]
        file_path = (
            self.get_abspath(self.paths[file_info["path"]]) / file_info["filename"]
        )

        if not file_path.exists():
            self.download_file(file_key)

        return file_path

    def get_folder_file_path(self, folder, filename, default=None):
        return self.get_file_path(f"{folder}/{filename}", default=default)

    def download_file(self, file_key):
        """
        Download a file if it doesn't exist.
        """
        file_info = self.DOWNLOADABLE_FILES[file_key]
        file_path = (
            self.get_abspath(self.paths[file_info["path"]]) / file_info["filename"]
        )

        print(f"Downloading {file_info['url']}...")
        response = requests.get(file_info["url"], stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(file_path, "wb") as file, tqdm(
            desc=file_info["filename"],
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

        print(f"Downloaded {file_info['filename']} to {file_path}")

    def find_lcm_lora(self):
        return self.get_file_path("lcm_lora")
