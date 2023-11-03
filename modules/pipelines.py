import modules.lcm_pipeline as lcm_pipeline
import modules.sdxl_pipeline as sdxl_pipeline
from shared import state

def update(gen_data):
    try:
        if gen_data["base_model_name"].startswith("lcm/"):
            if state["pipeline"] is None or "lcm" not in state["pipeline"].pipeline_type:
                state["pipeline"] = lcm_pipeline.pipeline()
        else:
            if state["pipeline"] is None or "sdxl" not in state["pipeline"].pipeline_type:
                state["pipeline"] = sdxl_pipeline.pipeline()
        return state["pipeline"]
    except:
        return None
