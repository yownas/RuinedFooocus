import os
from shared import state, path_manager
from modules.civit import Civit
from pathlib import Path

civit = Civit(cache_path=Path(path_manager.model_paths["cache_path"]) / Path("checkpoints"))

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

class NoPipeLine:
    pipeline_type = []

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
            baseModel = None
            if "base_model_name" in gen_data:
                file = Path(path_manager.model_paths["modelfile_path"]) / Path(gen_data['base_model_name'])
                baseModel = civit.get_model_base(civit.get_models_by_path(file))
            if state["pipeline"] is None:
                state["pipeline"] = NoPipeLine()

            if baseModel is not None:
                # Try with SDXL if we have an "Unknown" model.
                if (
                    baseModel in ["Playground v2", "Pony", "SD 3", "SDXL 1.0", "SDXL Distilled", "SDXL Hyper", "SDXL Turbo", "Unknown", "Merge"]
                    and "sdxl" not in state["pipeline"].pipeline_type
                ):
                    state["pipeline"] = sdxl_pipeline.pipeline()

        if state["pipeline"] is None or len(state["pipeline"].pipeline_type) == 0:
            print(f"Warning: Using SDXL pipeline as fallback.")
            state["pipeline"] = sdxl_pipeline.pipeline()

        return state["pipeline"]
    except:
        # If things fail. Use the template pipeline that only returns a logo
        state["pipeline"] = template_pipeline.pipeline()
        return state["pipeline"]
