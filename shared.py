try:
    from transformers import CLIPTokenizer
except:
    pass
from modules.settings import SettingsManager
from modules.performance import PerformanceSettings
from modules.resolutions import ResolutionSettings
from modules.path import PathManager
from modules.model_handler import Models

from argparser import args

gradio_root = None

state = {
    "preview_image": None,
    "ctrls_name": [],
    "ctrls_obj": [],
    "setting_name": [],
    "setting_obj": [],
    "cfg_items_name": [],
    "cfg_items_obj": [],
    "pipeline": None,
    "last_config": 0.0,
}

wildcards = None
try:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
except:
    print("No tokenizer in shared.py yet")
    tokenizer = None

settings = SettingsManager()
path_manager = PathManager()
performance_settings = PerformanceSettings()
resolution_settings = ResolutionSettings()
models = Models()
shared_cache = {}

def add_ctrl(name, obj, configurable=False):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]
    if configurable:
        state["cfg_items_name"] += [name]
        state["cfg_items_obj"] += [obj]

def add_setting(name, obj):
    state["setting_name"] += [name]
    state["setting_obj"] += [obj]

def add_cfg_item(name, obj):
    state["cfg_items_name"] += [name]
    state["cfg_items_obj"] += [obj]