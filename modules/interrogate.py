import torch
from clip_interrogator import Config, Interrogator
from PIL import Image
import json
from shared import path_manager, settings
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from modules.util import TimeIt

import os
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch

def brainblip_look(image, prompt, gr):
    from transformers import AutoProcessor, BlipForConditionalGeneration
    from PIL import Image

    gr.Info("BrainBlip is creating Your Prompt")
    print(f"Loading BrainBlip.")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("braintacles/brainblip").to("cpu")

    print(f"Processing...")
    inputs = processor(image, return_tensors="pt").to("cpu")
    out = model.generate(**inputs, min_length=40, max_new_tokens=150, num_beams=5, repetition_penalty=1.40)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

def clip_look(image, prompt, gr):
    text = "Lets interrogate"

    # Unload models, if needed?
    #state["pipeline"] = None
    gr.Info("Clip is reading Your Prompt")

    conf = Config(
        device=torch.device("cuda"),
        clip_model_name="ViT-L-14/openai",
        cache_path=path_manager.model_paths["clip_path"],
    )
    conf.apply_low_vram_defaults()

    i = Interrogator(conf)
    text = i.interrogate(image)

    return text

def florence_look(image, prompt, gr):
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        """Work around for https://huggingface.co/microsoft/Florence-2-large-ft/discussions/4 ."""
        if os.path.basename(filename) != "modeling_florence2.py":
            return get_imports(filename)
        imports = get_imports(filename)
        try:
            imports.remove("flash_attn")
        except:
            pass
        return imports

    text = "Lets interrogate"
    print(f"Looking...")
    image = image.convert('RGB')

    #state["pipeline"] = None
    gr.Info("Florence is creating Your Prompt")

    with TimeIt(""):
        device = "cpu"
        torch_dtype = torch.float32

        prompt = "<MORE_DETAILED_CAPTION>"

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = Florence2ForConditionalGeneration.from_pretrained(
                "florence-community/Florence-2-base",
                dtype=torch.bfloat16,
            ).to(device)
            processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.bfloat16)
        print(f"Judging...")
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            num_beams=6,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        text = result[prompt]

    return text

looks = {
    "brainblip": brainblip_look,
    "clip": clip_look,
    "florence": florence_look,
}

def look(image, prompt, gr):
    if prompt.strip(" :") in looks:
        text = looks[prompt.strip(" :")](image, prompt, gr)
    else:
        if prompt != "":
            return prompt
        try:
            info = image.info
            params = info.get("parameters", "")
            text = json.dumps(json.loads(params))
        except:
            # Default interrogator
            interrogator = settings.default_settings.get("interrogator", "florence")
            text = looks[interrogator](image, prompt, gr)

    return text
