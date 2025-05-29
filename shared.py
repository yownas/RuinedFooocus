try:
    from transformers import CLIPTokenizer
except:
    pass
from modules.settings import SettingsManager
from modules.performance import PerformanceSettings
from modules.resolutions import ResolutionSettings
from modules.path import PathManager
from modules.model_handler import Models
from modules.translation_manager import TranslationManager
from argparser import args
import time

gradio_root = None
server_app = None
local_url = "http://127.0.0.1:7860/"
share_url = "http://127.0.0.1:7860/"

translation_manager = TranslationManager()
translation_manager.set_language(args.language)
translate = translation_manager.translate

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

# Call this to trigger a refresh of ui components
def update_cfg():
    state["last_config"] = str(time.time())

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