import os

import json

from os.path import exists


DEFAULT_PATHS = {
    "path_checkpoints": "../models/checkpoints/",
    "path_loras": "../models/loras/",
    "path_controlnet": "../models/controlnet/",
    "path_outputs": "../outputs/",
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
temp_outputs_path = get_abspath(paths["path_outputs"])

os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = "sd_xl_base_1.0_0.9vae.safetensors"
default_refiner_model_name = "sd_xl_refiner_1.0_0.9vae.safetensors"
default_lora_name = "sd_xl_offset_example-lora_1.0.safetensors"
default_lora_weight = 0.5

model_filenames = []
lora_filenames = []

extensions = [".pth", ".ckpt", ".bin", ".safetensors"]


def get_model_filenames(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, dirs, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in [".pth", ".ckpt", ".bin", ".safetensors"]:
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return sorted(filenames, key=lambda x: -1 if os.sep in x else 1)


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    return


update_all_model_names()
