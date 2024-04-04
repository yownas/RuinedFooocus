import os
from shared import state

try:
    import modules.faceswapper_pipeline as faceswapper_pipeline

    print("INFO: Faceswap enabled")
    state["faceswap_loaded"] = True
except:
    state["faceswap_loaded"] = False
import modules.sdxl_pipeline as sdxl_pipeline
import modules.template_pipeline as template_pipeline
import modules.upscale_pipeline as upscale_pipeline
import modules.search_pipeline as search_pipeline
import modules.controlnet as controlnet

def update(gen_data):
    prompt = gen_data["prompt"] if "prompt" in gen_data else ""
    cn_settings = controlnet.get_settings(gen_data)
    cn_type = cn_settings["type"] if "type" in cn_settings else ""

    try:
        if prompt == "ruinedfooocuslogo":
            if (
                state["pipeline"] is None
                or "template" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = template_pipeline.pipeline()

        elif prompt.startswith("search:"):
            if (
                state["pipeline"] is None
                or "search" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = search_pipeline.pipeline()

        elif cn_type.lower() == "upscale":
            if (
                state["pipeline"] is None
                or "upscale" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = upscale_pipeline.pipeline()

        elif cn_type.lower() == "faceswap" and state["faceswap_loaded"]:
            if (
                state["pipeline"] is None
                or "faceswap" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = faceswapper_pipeline.pipeline()

        else:
            if (
                state["pipeline"] is None
                or "sdxl" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = sdxl_pipeline.pipeline()

        return state["pipeline"]
    except:
        # If things fail. Use the template pipeline that only returns a logo
        state["pipeline"] = template_pipeline.pipeline()
        return state["pipeline"]
