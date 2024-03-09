import os

#import modules.faceswapper_pipeline as faceswapper_pipeline
import modules.sdxl_pipeline as sdxl_pipeline
import modules.template_pipeline as template_pipeline
import modules.upscale_pipeline as upscale_pipeline
import modules.controlnet as controlnet
from shared import state

def update(gen_data):
    prompt = gen_data["prompt"] if "prompt" in gen_data else ""
    cn_settings = controlnet.get_settings(gen_data)
    cn_type = cn_settings["type"] if "type" in cn_settings else ""

    print(f"DEBUG: {cn_type}")

    try:
        if prompt == "ruinedfooocuslogo":
            if state["pipeline"] is None or "template" not in state["pipeline"].pipeline_type:
                state["pipeline"] = template_pipeline.pipeline()

        elif cn_type.lower() == "upscale":
            if state["pipeline"] is None or "upscale" not in state["pipeline"].pipeline_type:
                state["pipeline"] = upscale_pipeline.pipeline()

#        elif cn_selection == "Faceswap":
#            if state["pipeline"] is None or "faceswap" not in state["pipeline"].pipeline_type:
#                state["pipeline"] = faceswapper_pipeline.pipeline()

        else:
            if state["pipeline"] is None or "sdxl" not in state["pipeline"].pipeline_type:
                state["pipeline"] = sdxl_pipeline.pipeline()

        return state["pipeline"]
    except:
        return None
