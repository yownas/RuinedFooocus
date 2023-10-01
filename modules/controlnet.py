canny_model = "control-lora-canny-rank128.safetensors"
depth_model = "control-lora-depth-rank128.safetensors"

# https://huggingface.co/stabilityai/control-lora/tree/main/control-LoRAs-rank128

controlnet_settings = {
    "Canny (low)": {
        "type": "canny",
        "model": canny_model,
        "edge_low": 0.2,
        "edge_high": 0.8,
        "strength": 0.5,
        "start": 0.0,
        "stop": 0.5
    },
    "Canny (high)": {
        "type": "canny",
        "model": canny_model,
        "edge_low": 0.2,
        "edge_high": 0.8,
        "strength": 1.0,
        "start": 0.0,
        "stop": 0.99
    },
    "Depth (low)": {
        "type": "depth",
        "model": depth_model,
        "strength": 0.5,
        "start": 0.0,
        "stop": 0.5
    },
    "Depth (high)": {
        "type": "depth",
        "model": depth_model,
        "strength": 1.0,
        "start": 0.0,
        "stop": 0.99
    },
}

def modes():
    return controlnet_settings.keys()

def get_settings(controlnet):
    return controlnet_settings[controlnet]

