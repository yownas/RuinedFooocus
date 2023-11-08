import os

import modules.faceswapper_pipeline as faceswapper_pipeline
import modules.lcm_pipeline as lcm_pipeline
import modules.sdxl_pipeline as sdxl_pipeline
import modules.template_pipeline as template_pipeline

import modules.controlnet
from shared import state

def update(gen_data):
    prompt = gen_data["prompt"] if "prompt" in gen_data else ""
    cn_selection = gen_data["cn_selection"] if "cn_selection" in gen_data else ""

    try:
        if prompt == "ruinedfooocuslogo":
            if state["pipeline"] is None or "template" not in state["pipeline"].pipeline_type:
                state["pipeline"] = template_pipeline.pipeline()

        elif cn_selection == "Faceswap":
            if state["pipeline"] is None or "faceswap" not in state["pipeline"].pipeline_type:
                state["pipeline"] = faceswapper_pipeline.pipeline()

        elif os.path.split(os.path.split(gen_data["base_model_name"])[0])[1].lower() == "lcm":
            if state["pipeline"] is None or "lcm" not in state["pipeline"].pipeline_type:
                state["pipeline"] = lcm_pipeline.pipeline()

        else:
            if state["pipeline"] is None or "sdxl" not in state["pipeline"].pipeline_type:
                state["pipeline"] = sdxl_pipeline.pipeline()

        return state["pipeline"]
    except:
        return None
