import os
import sys
import version
import warnings
from pathlib import Path
import ssl
import json
import shared
import torchruntime

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="kornia")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchsde")
warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from modules.launch_util import (
    is_installed,
    run,
    python,
    run_pip,
    pip_rm,
    repo_dir,
    requirements_met,
    script_path,
    dir_repos,
)


git_repos = [
    {
        "name": "ComfyUI",
        "path": "ComfyUI",
        "url": "https://github.com/comfyanonymous/ComfyUI",
        "hash": "b779349b55e79aff81a98b752f5cb486c71812db",
        "add_path": "ComfyUI",
    },
    {
        "name": "Calcuis-GGUF",
        "path": "calcuis_gguf",
        "url": "https://github.com/calcuis/gguf",
        "hash": "ea10dbe3d4c3ca3b18320315b322e401a6f72745",
        "add_path": "",
    },
]


def prepare_environment(offline=False):
    print(f"Python {sys.version}")
    print(f"RuinedFooocus version: {version.version}")

    requirements_file = "requirements_versions.txt"
    pip_config = "settings/pip.json"
    pip_data = {
        "setup": {"torch": "cu124", "llama": "cpu"},
        "installed": {"torch": "cu124", "llama": "cpu"},
    }
    try:
        with open(pip_config) as f:
            pip_data.update(json.load(f))
    except:
        print(f"INFO: Could not read setup file {pip_config}, using defaults.")
        pass

    modules_file = "pip/modules.txt"
    llama_file = f"pip/llama_{pip_data['setup']['llama']}.txt"

    if offline:
        print("Skip pip check.")
    else:
        run(
            f'"{python}" -m pip install --upgrade pip',
            "Check pip",
            "Couldn't check pip",
            live=False,
        )
        run(
            f'"{python}" -m pip install -r "{requirements_file}"',
            "Check pre-requirements",
            "Couldn't check pre-reqs",
            live=False,
        )

    if pip_data["setup"]["llama"] != pip_data["installed"]["llama"]:
        pip_rm("llama_cpp_python", "llama")

    if offline:
        print("Skip check of required modules.")
    else:
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

        # Run TorchUtils
        #run(
        #    f'"{python}" -m torchruntime install',
        #    "Checking for latest torch version",
        #    "Couldn't install torch on this machine",
        #    live=True,
        #)
        torchruntime.install()

        if REINSTALL_ALL or not requirements_met(modules_file):
            print("This next step may take a while")
            run_pip(f'install -r "{modules_file}"', "required modules")

        if REINSTALL_ALL or not is_installed("llama_cpp"):
            run_pip(f'install -r "{llama_file}"', "llama modules")
            pip_data["installed"]["llama"] = pip_data["setup"]["llama"]

    # Update pip.json
    try:
        with open(pip_config, "w") as file:
            json.dump(pip_data, file, indent=2)
    except:
        print(f"WARNING: Could not write setup file {pip_config}")
        pass
    shared.shared_cache["installed"] = pip_data["installed"]



def clone_git_repos(offline=False):
    from modules.launch_util import git_clone

    for repo in git_repos:
        if not offline:
            git_clone(repo["url"], repo_dir(repo["path"]), repo["name"], repo["hash"])
        add_path = str(Path(script_path) / dir_repos / repo["add_path"])
        if add_path not in sys.path:
            sys.path.append(add_path)


def download_models():
    from modules.util import load_file_from_url
    from shared import path_manager

    model_filenames = [
        (
            path_manager.model_paths["vae_approx_path"],
            "taesdxl_decoder",
            "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
        ),
        (
            "prompt_expansion",
            "pytorch_model.bin",
            "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        ),
    ]

    for model_dir, file_name, url in model_filenames:
        load_file_from_url(
            url=url,
            model_dir=model_dir,
            file_name=file_name,
        )


from argparser import args

REINSTALL_ALL = False
if os.path.exists("reinstall"):
    REINSTALL_ALL = True

if args.gpu_device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

offline = os.environ.get("RF_OFFLINE") == "1" or "--offline" in sys.argv

if offline:
    print("Skip checking python modules.")

prepare_environment(offline)

if os.path.exists("reinstall"):
    os.remove("reinstall")

try:
    clone_git_repos(offline)
except:
    print(f"WARNING: Failed checking git-repos. Trying to start without update.")

if not offline:
    download_models()

torchruntime.configure()
print("Starting webui")
from webui import *
