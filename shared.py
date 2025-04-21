try:
    from transformers import CLIPTokenizer
except:
    pass
from modules.settings import SettingsManager
from modules.performance import PerformanceSettings
from modules.resolutions import ResolutionSettings
from modules.path import PathManager
from modules.model_handler import Models

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

path_manager = PathManager()
settings = SettingsManager()
performance_settings = PerformanceSettings()
resolution_settings = ResolutionSettings()
models = Models()

shared_cache = {}

def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]

def add_setting(name, obj):
    state["setting_name"] += [name]
    state["setting_obj"] += [obj]

