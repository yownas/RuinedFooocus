import os
import sys
import platform
import version
import warnings
from pathlib import Path
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
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
    requirements_met,
    script_path,
    dir_repos,
)

comfy_repo = (
    os.environ.get("COMFY_REPO", "https://github.com/comfyanonymous/ComfyUI"),
    os.environ.get("COMFY_COMMIT_HASH", "935ae153e154813ace36db4c4656a5e96f403eba"),
)
sf3d_repo = (
    os.environ.get("SF3D_REPO", "https://github.com/Stability-AI/stable-fast-3d.git"),
    os.environ.get("SF3D_COMMIT_HASH", "070ece138459e38e1fe9f54aa19edb834bced85e"),
)

REINSTALL_ALL = False
if os.path.exists("reinstall"):
    REINSTALL_ALL = True

def prepare_environment():
    torch_index_url = os.environ.get(
        "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121"
    )
    torch_command = os.environ.get(
        "TORCH_COMMAND",
        f"pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url {torch_index_url}",
    )
    requirements_file = os.environ.get("REQS_FILE", "requirements_versions.txt")

    xformers_package = os.environ.get("XFORMERS_PACKAGE", "xformers==0.0.26")

    print(f"Python {sys.version}")
    print(f"RuinedFooocus version: {version.version}")

    run(f'"{python}" -m pip install --upgrade pip', "Check pip", "Couldn't check pip", live=True)

    if not is_installed("wheel"):
        run(f'"{python}" -m pip install wheel', "Installing wheel", "Couldn't install wheel", live=True)

    if not is_installed("packaging"):
        run(f'"{python}" -m pip install packaging', "Installing packaging", "Couldn't install packaging", live=True)

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
        print("This next step may take a while")
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
        run_pip(f'install -r "{requirements_file}" --extra-index-url {torch_index_url}', "requirements")

    return

def clone_git_repos():
    from modules.launch_util import git_clone

    comfyui_name = "ComfyUI-from-StabilityAI-Official"
    git_clone(comfy_repo[0], repo_dir(comfyui_name), "Comfy Backend", comfy_repo[1])
    path = Path(script_path) / dir_repos / comfyui_name
    sys.path.append(str(path))

    sf3d_name = "stable-fast-3d"
    git_clone(sf3d_repo[0], repo_dir(sf3d_name), "Stable Fast 3D", sf3d_repo[1])
    path = Path(script_path) / dir_repos / "stable-fast-3d"
    sys.path.append(str(path))

    return


def download_models():
    from modules.util import load_file_from_url
    from shared import path_manager

    model_filenames = [
        (
            path_manager.model_paths["modelfile_path"],
            "sd_xl_base_1.0_0.9vae.safetensors",
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
        ),
        (
            path_manager.model_paths["lorafile_path"],
            "sd_xl_offset_example-lora_1.0.safetensors",
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors",
        ),
        (
            path_manager.model_paths["lorafile_path"],
            "lcm-lora-sdxl.safetensors",
            "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
        ),
        (
            path_manager.model_paths["lorafile_path"],
            "lcm-lora-ssd-1b.safetensors",
            "https://huggingface.co/latent-consistency/lcm-lora-ssd-1b/resolve/main/pytorch_lora_weights.safetensors",
        ),
        (
            path_manager.model_paths["vae_approx_path"],
            "taesdxl_decoder",
            "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth",
        ),
        (
            path_manager.model_paths["controlnet_path"],
            "control-lora-canny-rank128.safetensors",
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors",
        ),
        (
            path_manager.model_paths["controlnet_path"],
            "control-lora-depth-rank128.safetensors",
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors",
        ),
        (
            path_manager.model_paths["controlnet_path"],
            "control-lora-recolor-rank128.safetensors",
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors",
        ),
        (
            path_manager.model_paths["controlnet_path"],
            "control-lora-sketch-rank128-metadata.safetensors",
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors",
        ),
        (
            path_manager.model_paths["upscaler_path"],
            "4x-UltraSharp.pth",
            "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        ),
        (
            "prompt_expansion",
            "pytorch_model.bin",
            "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        ),
        (
            "models/layerdiffuse/",
            "layer_xl_transparent_attn.safetensors",
            "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
        ),
        (
            "models/layerdiffuse/",
            "vae_transparent_decoder.safetensors",
            "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
        ),
    ]

    for model_dir, file_name, url in model_filenames:
        load_file_from_url(
            url=url,
            model_dir=model_dir,
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
clone_git_repos()

clear_comfy_args()
# cuda_malloc()

download_models()

from webui import *
