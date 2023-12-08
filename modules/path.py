import os
from pathlib import Path
import json

from os.path import exists


DEFAULT_PATHS = {
    "path_checkpoints": "../models/checkpoints/",
    "path_loras": "../models/loras/",
    "path_controlnet": "../models/controlnet/",
    "path_vae_approx": "../models/vae_approx/",
    "path_preview": "../outputs/preview.jpg",
    "path_upscalers": "../models/upscale_models",
    "path_faceswap": "../models/faceswap",
    "path_outputs": "../outputs/",
    "path_clip": "../models/clip/",
}


def load_paths():
    paths = DEFAULT_PATHS.copy()

    if exists("settings/paths.json"):
        with open("settings/paths.json") as f:
            paths.update(json.load(f))

    for key in DEFAULT_PATHS:
        if key not in paths:
            paths[key] = DEFAULT_PATHS[key]

    with open("settings/paths.json", "w") as f:
        json.dump(paths, f, indent=2)

    return paths


paths = load_paths()


def get_abspath(path):
    return (
        path
        if os.path.isabs(path)
        else os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    )


modelfile_path = get_abspath(paths["path_checkpoints"])
lorafile_path = get_abspath(paths["path_loras"])
controlnet_path = get_abspath(paths["path_controlnet"])
vae_approx_path = get_abspath(paths["path_vae_approx"])
temp_outputs_path = get_abspath(paths["path_outputs"])
temp_preview_path = get_abspath(paths["path_preview"])
upscaler_path = get_abspath(paths["path_upscalers"])
faceswap_path = get_abspath(paths["path_faceswap"])
clip_path = get_abspath(paths["path_clip"])

os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = "sd_xl_base_1.0_0.9vae.safetensors"
default_lora_name = "sd_xl_offset_example-lora_1.0.safetensors"
default_lora_weight = 0.5

model_filenames = []
lora_filenames = []
upscaler_filenames = []

extensions = [".pth", ".ckpt", ".bin", ".safetensors"]


def get_model_filenames(folder_path, isLora=False):
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for path in folder_path.rglob("*"):
        if path.suffix.lower() in [".pth", ".ckpt", ".bin", ".safetensors"]:
            if isLora:
                txtcheck = path.with_suffix(".txt")
                if txtcheck.exists():
                    path = path.with_suffix(f"{path.suffix} 🗒️")

            filenames.append(str(path.relative_to(folder_path)))

    return sorted(
        filenames,
        key=lambda x: f"0{x.casefold()}" if Path(x).suffix in x else f"1{x.casefold()}",
    )


def update_all_model_names():
    global model_filenames, lora_filenames, upscaler_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path, True)
    upscaler_filenames = get_model_filenames(upscaler_path)
    return


def find_lcm_lora():
    path = lorafile_path
    filename = "lcm-lora-sdxl.safetensors"
    for root, dirs, files in os.walk(path):
        if filename in files:
            return os.path.relpath(os.path.join(root, filename), path)


update_all_model_names()
