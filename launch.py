import os
import sys
import version
import warnings
from pathlib import Path
import ssl
from tempfile import gettempdir

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

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
        "hash": "bd951a714f8c736680fe13e735eee71acf73dd4c",
        "add_path": "ComfyUI",
    },
    {
        "name": "Calcuis-GGUF",
        "path": "calcuis_gguf",
        "url": "https://github.com/calcuis/gguf",
        "hash": "2504e21db88e1b5427ce358dbf206f3b3e6fad80",
        "add_path": "",
    },
]


def prepare_environment(offline=False):
    print(f"Python {sys.version}")
    print(f"RuinedFooocus version: {version.version}")

    requirements_file = "requirements_versions.txt"

    modules_file = "pip/modules.txt"

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
        run(
            f'"{python}" -m pip uninstall -y llama-cpp-python',
            "Check for old modules",
            "Couldn't check old modules",
            live=False,
        )


    import torchruntime
    import platform
    gpus = torchruntime.device_db.get_gpus()
    if "TORCH_PLATFORM" in os.environ:
        torch_platform = os.environ["TORCH_PLATFORM"]
    else:
        torch_platform = torchruntime.platform_detection.get_torch_platform(gpus)
    os_platform = platform.system()

    # Some platform checks
    if torch_platform == "xpu" and not os_platform == "Windows":
        torch_platform == "cpu"
    if torch_platform == "mps" and not os_platform == "Darwin":
        torch_platform == "cpu"

    print(f"Torch platform: {os_platform}: {torch_platform}") # Some debug output

    if offline:
        print("Skip check of required modules.")
    else:
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

        # Run torchruntime install
        cmds = torchruntime.installer.get_install_commands(torch_platform, [])
        cmds = torchruntime.installer.get_pip_commands(cmds)
        torchruntime.installer.run_commands(cmds)
        torchruntime.configure()

        if REINSTALL_ALL or not requirements_met(modules_file):
            print("This next step may take a while")
            run_pip(f'install -r "{modules_file}"', "required modules")

        if REINSTALL_ALL or not is_installed("nexa"):
            platform_index = {
                'cu124': 'https://github.nexa.ai/whl/cu124',
                'cu128': 'https://github.nexa.ai/whl/cu124',
                'rocm6.2': 'https://github.nexa.ai/whl/rocm621',
                'directml': 'https://github.nexa.ai/whl/vulkan',
                'mps': 'https://github.nexa.ai/whl/metal',
                'xpu': 'https://github.nexa.ai/whl/sycl',
                'cpu': 'https://github.nexa.ai/whl/cpu'
            }
            if torch_platform in platform_index:
                run_pip(f'install nexaai -U --extra-index-url {platform_index[torch_platform]}', "Nexa SDK modules")
            else:
                print(f"ERROR: Can't find Nexai SDK url for {torch_platform}")


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

offline = os.environ.get("RF_OFFLINE") == "1" or "--offline" in sys.argv or "--iINSTallLEDmYOwNPaCKaGeS" in sys.argv

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

gradio_cache = os.path.join(gettempdir(), 'ruinedfooocus_cache')
os.environ['GRADIO_TEMP_DIR'] = gradio_cache
# Delete old data
import shutil
try:
    # Yownas being paranoid
    if gradio_cache.endswith('ruinedfooocus_cache'):
        shutil.rmtree(gradio_cache)
except FileNotFoundError:
    pass
except PermissionError:
    pass
except Exception as e:
    print(f"An error occurred: {str(e)}")

def launch_ui():
    print("Starting webui")
    import webui
launch_ui()
