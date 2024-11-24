import datetime
import random
import time
from pathlib import Path

from contextlib import contextmanager
from typing import Optional
from urllib.parse import urlparse
from shared import path_manager, shared_cache
import json


def get_wildcard_files():
    directories = ["wildcards", "wildcards_official"]
    files = []

    for directory in directories:
        for file in Path(directory).rglob("*.txt"):
            name = file.stem
            if name not in files:
                files.append(name)

    onebutton = [
        "onebuttonprompt",
        "onebuttonsubject",
        "onebuttonhumanoid",
        "onebuttonmale",
        "onebuttonfemale",
        "onebuttonanimal",
        "onebuttonobject",
        "onebuttonlandscape",
        "onebuttonconcept",
        "onebuttonartist",
        "onebutton1girl",
        "onebutton1boy",
        "onebuttonfurry",
    ]
    both = files + onebutton
    return both


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""
    try:
        with open(filename, "rb") as file:
            import hashlib

            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            shorthash = m.hexdigest()[0:8]
            return shorthash
    except FileNotFoundError:
        return "NOFILE"
    except Exception:
        return "NOHASH"


def generate_temp_filename(folder="./outputs/", extension="png"):
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    random_number = random.randint(1000, 9999)
    filename = f"{time_string}_{random_number}.{extension}"
    result = Path(folder) / date_string / filename
    return result.absolute()


def load_keywords(lora):
    filename = Path(
        path_manager.model_paths["cache_path"] / "loras" / Path(lora).name
    ).with_suffix(".txt")
    try:
        with open(filename, "r") as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return " "

def _get_model_hashes(cache_path, not_found=None):
    hashes = {
        "AutoV1": "",
        "AutoV2": "",
        "SHA256": "",
        "CRC32": "",
        "BLAKE3": "",
        "AutoV3": ""
    }
    filename = cache_path.with_suffix(".json")
    if Path(filename).is_file():
        try:
            with open(filename) as f:
                data = json.load(f)
        except:
            print(f"ERROR: model {cache_path} is missing json-file")
            data = {}
        if "files" not in data:
            data = {"files": [{"hashes": {}}]}
        hashes.update(data['files'][0]['hashes'])
        return hashes
    else:
        if not_found:
            return not_found
        else:
            return hashes

def get_checkpoint_hashes(model):
    return _get_model_thumbnail(
        Path(path_manager.model_paths["cache_path"] / "checkpoints" / Path(model).name)
    )

def get_lora_hashes(model):
    return _get_model_hashes(
        Path(path_manager.model_paths["cache_path"] / "loras" / Path(model).name)
    )

def _get_model_thumbnail(cache_path, not_found="html/warning.png"):
    if cache_path in shared_cache:
        return shared_cache[cache_path]
    suffixes = [".jpeg", ".jpg", ".png", ".gif"]
    for suffix in suffixes:
        filename = cache_path.with_suffix(suffix)
        if Path(filename).is_file():
            shared_cache[cache_path] = str(filename)
            return str(filename)
    else:
        return not_found

def get_model_thumbnail(model):
    res = _get_model_thumbnail(
        Path(path_manager.model_paths["cache_path"] / "checkpoints" / Path(model).name),
        not_found=None
    )
    if res is not None:
        return res
    res = _get_model_thumbnail(
        Path(path_manager.model_paths["cache_path"] / "loras" / Path(model).name),
        not_found=None
    )
    if res is not None:
        return str(res)
    else:
        return "html/warning.png"

def get_checkpoint_thumbnail(model):
    if Path(model).suffix == ".merge":
        not_found="html/merge.jpeg"
    else:
        not_found="html/warning.jpeg"

    return _get_model_thumbnail(
        Path(path_manager.model_paths["cache_path"] / "checkpoints" / Path(model).name),
        not_found=not_found
    )


def get_lora_thumbnail(model):
    return _get_model_thumbnail(
        Path(path_manager.model_paths["cache_path"] / "loras" / Path(model).name)
    )

def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = Path(parts.path).stem

    for file in Path(model_dir).glob("**/*"):
        if file.name == file_name:
            cached_file = file
            return str(cached_file)

    cached_file = Path(model_dir) / file_name
    if not cached_file.exists():
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    return str(cached_file)


class TimeIt:
    def __init__(self, text=""):
        self.text = text

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"\033[91mTime taken: {self.interval:0.2f} seconds {self.text}\033[0m")


def remove_empty_str(items, default=None):
    items = [x for x in items if x != ""]
    if len(items) == 0 and default is not None:
        return [default]
    return items
