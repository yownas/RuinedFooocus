import datetime
import random
import os
import sys
import time
from pathlib import Path

from contextlib import contextmanager
from typing import Optional
from urllib.parse import urlparse


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
    filename = lora.replace(".safetensors", ".txt")
    try:
        with open(filename, "r") as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return " "


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
