import requests
import hashlib
import shutil
import os
import cv2
import json
import threading
import time
from pathlib import Path
import numpy as np

class Civit:
    def civit_update_worker(self, folder_path, cache_path):
        try:
            import imageio.v3
        except:
            # Skip updates if we are missing imageio
            print(f"DEBUG: Skip CivitAI update")
            return
        if folder_path in self.civit_worker_folders:
            # Already working on this folder
            return

        self.civit_worker_folders.append(folder_path)
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in self.EXTENSIONS:

                # get file name, add cache path change suffix
                cache_file = Path(cache_path / path.name)
                models = self.get_models_by_path(str(path))

                suffixes = [".jpeg", ".jpg", ".png", ".gif"]
                has_preview = False
                for suffix in suffixes:
                    thumbcheck = cache_file.with_suffix(suffix)
                    if Path(thumbcheck).is_file():
                        has_preview = True
                        break

                if not has_preview:
                    #print(f"Downloading model thumbnail for {Path(path).name} ({self.get_model_base(models)} - {self.get_model_type(models)})")
                    self.get_image(models, thumbcheck)
                    time.sleep(1)

                txtcheck = cache_file.with_suffix(".txt")
                if str(self.get_model_type(models)).lower() == "lora" and not txtcheck.exists():
                    print(f"Get LoRA keywords for {Path(path).name} ({self.get_model_base(models)} - {self.get_model_type(models)})")
                    keywords = self.get_keywords(models)
                    with open(txtcheck, "w") as f:
                        f.write(", ".join(keywords))

        self.civit_worker_folders.remove(folder_path)

    def __init__(self, model_dir=None, base_url="https://civitai.com/api/v1/", cache_path="cache"):
        self.model_dir = model_dir
        self.cache_path = cache_path
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.civit_worker_folders = []
        self.EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors", ".merge"]

        if model_dir:
            threading.Thread(
                target=self.civit_update_worker,
                args=(
                    self.model_dir,
                    self.cache_path,
                ),
                daemon=True,
            ).start()

    def _read_file(self, filename):
        try:
            with open(filename, "rb") as file:
                file.seek(0x100000)
                return file.read(0x10000)
        except FileNotFoundError:
            return b"NOFILE"
        except Exception:
            return b"NOHASH"

    def model_hash(self, filename):
        """old hash that only looks at a small part of the file and is prone to collisions"""
        file_content = self._read_file(filename)
        m = hashlib.sha256()
        m.update(file_content)
        shorthash = m.hexdigest()[0:8]
        return shorthash

    def model_sha256(self, filename):
        print(f"Hashing {filename}")
        blksize = 1024 * 1024
        hash_sha256 = hashlib.sha256()
        try:
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(blksize), b""):
                    hash_sha256.update(chunk)
            f.close()
            return hash_sha256.hexdigest().upper()
        except Exception as e:
            print(f"model_sha256(): Failed reading {filename}")
            print(f"Error: {e}")
            return None

    def get_models_by_path(self, path, cache_path=None):
        data = None
        if cache_path is None:
            cache_path = Path(self.cache_path) / Path(Path(path).name)
        json_path = Path(cache_path).with_suffix(".json")

        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except:
                data = None
        if data is not None:
            return data

        if Path(path).suffix == ".merge":
            return {"baseModel": "Merge"}

        hash = self.model_sha256(path)
        url = f"{self.base_url}model-versions/by-hash/{hash}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"Error: Model {Path(path).name} Not Found on civit.ai")
            elif response.status_code == 503:
                print("Error: Civit.ai Service Currently Unavailable")
            else:
                print(f"HTTP Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

        if data is None:
            data = {}

        print(f"Update model data: {json_path}")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return data

    def get_keywords(self, model):
        keywords = model.get("trainedWords", [""])
        return keywords

    def get_model_base(self, model):
        return model.get("baseModel", "Unknown")

    def get_model_type(self, model):
        res = model.get("model", None)
        if res is not None:
            res = res.get("type", "Unknown")
        else:
            res = "Unknown"
        return res

    def get_image(self, model, path):
        if "baseModel" in model and model["baseModel"] == "Merge":
            return

        import imageio.v3 as iio

        max = 166  # Max width or height

        def make_thumbnail(image, text, zoom=False, caption=False):
            from modules.settings import default_settings

            res = None

            if "model_preview" in default_settings:
                opts = default_settings["model_preview"].split(",")
                if "zoom" in opts:
                    zoom=True
                if "caption" in opts:
                    caption=True

            if zoom:
                oh = image.shape[0]
                ow = image.shape[1]
                scale = max / oh if oh > ow else max / ow
                res = cv2.resize(
                    image,
                    dsize=(int(ow * scale), int(oh * scale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if caption:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2

                org = (7, 32)
                color = (25, 15, 11) # BGR
                res = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )
                org = (5, 30)
                color = (243, 195, 165) # BGR
                res = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

            return res

        path = path.with_suffix(".jpeg")
        caption = f"{path.with_suffix('').name}"

        image_url = None
        for preview in model.get("images", [{}]):
            url = preview.get("url")
            format = preview.get("type")
            if url:
                print(f"Updating preview for {caption}.")
                image_url = url
                response = self.session.get(image_url)
                if response.status_code != 200:
                    print(f"WARNING: get_image() - {response.status_code} : {response.reason}")
                    break
                image = np.asarray(bytearray(response.content), dtype="uint8") 
                out = make_thumbnail(cv2.imdecode(image, cv2.IMREAD_COLOR), caption)
                if out is not None:
                    out = cv2.imencode('.jpg', out)[1] 
                else:
                    out = response.content
                with open(path, "wb") as file:
                    file.write(out)

                if format == "video":
                    tmp_path = f"{path}.tmp"
                    os.rename(path, tmp_path)
                    video = iio.imiter(tmp_path)
                    fps = iio.immeta(tmp_path)["fps"]
                    video_out = []
                    for i in video:
                        out = make_thumbnail(i, caption, zoom=True)
                        video_out.append(out)
                    iio.imwrite(
                        str(path.with_suffix(".gif")), video_out, fps=fps, loop=0
                    )
                    os.remove(tmp_path)
                break
        if image_url is None:
            shutil.copyfile("html/warning.jpeg", path)
