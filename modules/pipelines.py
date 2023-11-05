import modules.lcm_pipeline as lcm_pipeline
import modules.sdxl_pipeline as sdxl_pipeline
import modules.template_pipeline as template_pipeline

from shared import state

def update(gen_data):
    prompt = gen_data["prompt"] if "prompt" in gen_data else ""
    try:
        if prompt == "ruinedfooocus_test":
            if state["pipeline"] is None or "template" not in state["pipeline"].pipeline_type:
                state["pipeline"] = template_pipeline.pipeline()

        elif gen_data["base_model_name"].startswith("lcm/"):
            if state["pipeline"] is None or "lcm" not in state["pipeline"].pipeline_type:
                state["pipeline"] = lcm_pipeline.pipeline()

        else:
            if state["pipeline"] is None or "sdxl" not in state["pipeline"].pipeline_type:
                state["pipeline"] = sdxl_pipeline.pipeline()
        return state["pipeline"]
    except:
        return None
