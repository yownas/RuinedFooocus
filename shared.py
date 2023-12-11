from transformers import CLIPTokenizer
from modules.performance import PerformanceSettings

gradio_root = None

state = {"preview_image": None, "ctrls_name": [], "ctrls_obj": [], "pipeline": None}

wildcards = None
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

performance_settings = PerformanceSettings()


def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]
