try:
    from transformers import CLIPTokenizer
except:
    pass
from modules.performance import PerformanceSettings
from modules.resolutions import ResolutionSettings
from modules.path import PathManager

gradio_root = None

state = {
    "preview_image": None,
    "ctrls_name": [],
    "ctrls_obj": [],
    "setting_name": [],
    "setting_obj": [],
    "pipeline": None,
}

wildcards = None
try:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
except:
    print("No tokenizer in shared.py")
    tokenizer = None

performance_settings = PerformanceSettings()
resolution_settings = ResolutionSettings()
civit_workers = []
path_manager = PathManager()
models = None

shared_cache = {}

def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]

def add_setting(name, obj):
    state["setting_name"] += [name]
    state["setting_obj"] += [obj]

