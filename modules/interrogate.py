import torch
import clip_interrogator as ci
from PIL import Image
import json
from shared import state
from modules.path import clip_path


def look(image):
    try:
        info = image.info
        params = info.get("parameters", "")
        text = json.loads(params)
    except:
        text = "Lets interrogate"

        # Unload models, if needed?
        # state["pipeline"] = None

        conf = ci.Config(
            device=torch.device("cuda"),
            clip_model_name="ViT-L-14/openai",
            cache_path=clip_path,
        )
        conf.apply_low_vram_defaults()

        i = ci.Interrogator(conf)
        text = i.interrogate(image)

    return text
