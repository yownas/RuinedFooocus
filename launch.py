import os
import sys
import platform
import version
import warnings
from pathlib import Path
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
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
    repo_dir,
    git_clone,
    requirements_met,
    script_path,
    dir_repos,
)
from modules.util import load_file_from_url
from shared import path_manager

REINSTALL_ALL = False
if os.path.exists("reinstall"):
    REINSTALL_ALL = True


def prepare_environment():
    torch_index_url = os.environ.get(
        "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121"
    )
    torch_command = os.environ.get(
        "TORCH_COMMAND",
        f"pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url {torch_index_url}",
    )
    insightface_package = os.environ.get(
        "INSIGHTFACE_PACKAGE",
        f"https://github.com/Gourieff/sd-webui-reactor/raw/main/example/insightface-0.7.3-cp310-cp310-win_amd64.whl",
    )
    requirements_file = os.environ.get("REQS_FILE", "requirements_versions.txt")

    xformers_package = os.environ.get("XFORMERS_PACKAGE", "xformers==0.0.23")

    comfy_repo = os.environ.get(
        "COMFY_REPO", "https://github.com/comfyanonymous/ComfyUI"
    )
    comfy_commit_hash = os.environ.get(
        "COMFY_COMMIT_HASH", "30abc324c2f73e6b648093ccd4741dece20be1e5"
    )

    print(f"Python {sys.version}")
    print(f"RuinedFooocus version: {version.version}")

    comfyui_name = "ComfyUI-from-StabilityAI-Official"
    git_clone(comfy_repo, repo_dir(comfyui_name), "Comfy Backend", comfy_commit_hash)
    path = Path(script_path) / dir_repos / comfyui_name
    sys.path.append(str(path))

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(
            f'"{python}" -m {torch_command}',
            "Installing torch and torchvision",
            "Couldn't install torch",
            live=True,
        )

    if REINSTALL_ALL or not is_installed("xformers"):
        if platform.system() == "Windows":
            if platform.python_version().startswith("3.10"):
                run_pip(
                    f"install -U -I --no-deps {xformers_package}", "xformers", live=True
                )
            else:
                print(
                    "Installation of xformers is not supported in this version of Python."
                )
                print(
                    "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness"
                )
                if not is_installed("xformers"):
                    exit(0)
        elif platform.system() == "Linux":
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements")

    return


model_filenames = [
    (
        "sd_xl_base_1.0_0.9vae.safetensors",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
    ),
]

lora_filenames = [
    (
        "sd_xl_offset_example-lora_1.0.safetensors",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors",
    ),
    (
        "lcm-lora-sdxl.safetensors",
        "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
    ),
    (
        "lcm-lora-ssd-1b.safetensors",
        "https://huggingface.co/latent-consistency/lcm-lora-ssd-1b/resolve/main/pytorch_lora_weights.safetensors",
    ),
]

vae_approx_filenames = [
    (
        "taesdxl_decoder",
        "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
    )
]

controlnet_filenames = [
    (
        "control-lora-canny-rank128.safetensors",
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors",
    ),
    (
        "control-lora-depth-rank128.safetensors",
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors",
    ),
    (
        "control-lora-recolor-rank128.safetensors",
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors",
    ),
    (
        "control-lora-sketch-rank128-metadata.safetensors",
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors",
    ),
]

upscaler_filenames = [
    (
        "4x-UltraSharp.pth",
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
    ),
]

magic_prompt_filenames = [
    (
        "pytorch_model.bin",
        "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
    ),
]

layer_diffuse_filenames = [
    (
        "layer_xl_transparent_attn.safetensors",
        "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
    ),
    (
        "vae_transparent_decoder.safetensors",
        "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
    ),
]


def download_models():
    for file_name, url in model_filenames:
        load_file_from_url(
            url=url,
            model_dir=path_manager.model_paths["modelfile_path"],
            file_name=file_name,
        )
    for file_name, url in lora_filenames:
        load_file_from_url(
            url=url,
            model_dir=path_manager.model_paths["lorafile_path"],
            file_name=file_name,
        )
    for file_name, url in controlnet_filenames:
        load_file_from_url(
            url=url,
            model_dir=path_manager.model_paths["controlnet_path"],
            file_name=file_name,
        )
    for file_name, url in vae_approx_filenames:
        load_file_from_url(
            url=url,
            model_dir=path_manager.model_paths["vae_approx_path"],
            file_name=file_name,
        )
    for file_name, url in upscaler_filenames:
        load_file_from_url(
            url=url,
            model_dir=path_manager.model_paths["upscaler_path"],
            file_name=file_name,
        )
    for file_name, url in magic_prompt_filenames:
        load_file_from_url(
            url=url,
            model_dir="prompt_expansion",
            file_name=file_name,
        )
    for file_name, url in layer_diffuse_filenames:
        load_file_from_url(
            url=url,
            model_dir="models/layerdiffuse/",
            file_name=file_name,
        )
    return


def clear_comfy_args():
    argv = sys.argv
    sys.argv = [sys.argv[0]]
    import comfy.cli_args

    sys.argv = argv


def cuda_malloc():
    import cuda_malloc


prepare_environment()
if os.path.exists("reinstall"):
    os.remove("reinstall")

clear_comfy_args()
# cuda_malloc()

download_models()

from webui import *
