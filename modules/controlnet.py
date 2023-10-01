import os
import json
from os.path import exists

controlnet_models = {
    "canny": "control-lora-canny-rank128.safetensors",
    "depth": "control-lora-depth-rank128.safetensors",
}

# https://huggingface.co/stabilityai/control-lora/tree/main/control-LoRAs-rank128

controlnet_settings = {
    "Canny (low)": {
        "type": "canny",
        "edge_low": 0.2,
        "edge_high": 0.8,
        "strength": 0.5,
        "start": 0.0,
        "stop": 0.5
    },
    "Canny (high)": {
        "type": "canny",
        "edge_low": 0.2,
        "edge_high": 0.8,
        "strength": 1.0,
        "start": 0.0,
        "stop": 0.99
    },
    "Depth (low)": {
        "type": "depth",
        "strength": 0.5,
        "start": 0.0,
        "stop": 0.5
    },
    "Depth (high)": {
        "type": "depth",
        "strength": 1.0,
        "start": 0.0,
        "stop": 0.99
    },
}

def load_settings():
    jsonfile = "controlnet.json"
    settings = controlnet_settings
    if exists(jsonfile):
        with open(jsonfile) as f:
            settings.update(json.load(f))
    else:
        with open(jsonfile, "w") as f:
            json.dump(settings, f, indent=2)
    return settings

controlnet_settings = load_settings()

def modes():
    return controlnet_settings.keys()

def get_model(type):
    return controlnet_models[type]

def get_settings(controlnet):
    return controlnet_settings[controlnet]

