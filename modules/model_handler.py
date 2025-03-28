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
from shared import civit_workers, path_manager
from modules.path import PathManager

class Models:

    def civit_update_worker(self, model_type, folder_paths):
        try:
            import imageio.v3
        except:
            # Skip updates if we are missing imageio
            print(f"Can't find imageio.v3 module: Skip CivitAI update")
            return
        if str(model_type) in civit_workers:
            # Already working on this folder
            print(f"Skip CivitAI check. Update for {model_type} already running.")
            return
        if not Path(self.cache_paths[model_type]).is_dir():
            print(f"WARNING: Can't find {self.cache_paths[model_type]}  Will not update thumbnails.")
            return

        civit_workers.append(str(model_type))
        self.ready[model_type] = False
        updated = 0

        # Quick list
        self.names[model_type] = []
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    # Add to model names
                    self.names[model_type].append(str(path.relative_to(folder)))

        # Return a sorted list, prepend names with 0 if they are in a folder or 1
        # if it is a plain file. This will sort folders above files in the dropdown
        self.names[model_type] = sorted(
            self.names[model_type],
            key=lambda x: (
                f"0{x.casefold()}"
                if not str(Path(x).parent) == "."
                else f"1{x.casefold()}"
            ),
        )
        self.ready[model_type] = True

        if self.offline:
            civit_workers.remove(str(model_type))
            return

        if model_type == "inbox" and self.names["inbox"]:
            checkpoints = path_manager.model_paths["modelfile_path"]
            checkpoints = checkpoints[0] if isinstance(checkpoints, list) else checkpoints
            loras = path_manager.model_paths["lorafile_path"]
            loras = loras[0] if isinstance(loras, list) else loras
            folders = {
                "LORA": (loras, self.cache_paths["loras"]),
                "LoCon": (loras, self.cache_paths["loras"]),
                "Checkpoint": (checkpoints, self.cache_paths["checkpoints"]),
            }

        # Go though and check previews
        for folder in folder_paths:
            for path in folder.rglob("*"):
                if path.suffix.lower() in self.EXTENSIONS:
                    # get file name, add cache path change suffix
                    cache_file = Path(self.cache_paths[model_type] / path.name)
                    models = self.get_models_by_path(model_type, str(path))

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
                        updated += 1
                        time.sleep(1)

                    txtcheck = cache_file.with_suffix(".txt")
                    if model_type == "loras" and not txtcheck.exists():
                        print(f"Get LoRA keywords for {Path(path).name} ({self.get_model_base(models)} - {self.get_model_type(models)})")
                        keywords = self.get_keywords(models)
                        with open(txtcheck, "w") as f:
                            f.write(", ".join(keywords))
                        updated += 1

                    if model_type == "inbox" and self.names["inbox"]:
                        name = str(path.relative_to(folder_paths[0])) # FIXME inbox is a single folder, not a list
                        model = self.get_models_by_path("inbox", name)
                        filename =  self.get_file_from_name("inbox", name)
                        if model is None:
                            continue
                        baseModel = self.get_model_base(model)
                        folder, cache = folders.get(self.get_model_type(model), [None, None])
                        if folder is None or baseModel is None:
                            print(f"Skipping {name} not sure what {self.get_model_type(model)} is.")
                            continue
                        # Move model to correct folder
                        dest = Path(folder) / baseModel
                        if not dest.exists():
                            dest.mkdir(parents=True, exist_ok=True)
                        Path(filename).rename(dest / name)
                        # Move cache-files
                        cache_file = Path(self.cache_paths[model_type] / name)
                        suffixes = [".json", ".txt", ".jpeg", ".jpg", ".png", ".gif"]
                        for suffix in suffixes:
                            cachefile = cache_file.with_suffix(suffix)
                            if cachefile.is_file():
                                cachefile.rename(cache / cachefile.name)
                        print(f"Moved {name} to {dest}")

        if updated > 0:
            print(f"CivitAI update for {model_type} done.")
        civit_workers.remove(str(model_type))

    def get_names(self, model_type):
        while not self.ready[model_type]:
            # Wait until we have read all the filenames
            time.sleep(0.2)
        return self.names[model_type]

    def get_file(self, model_type, name):
        # Search the folders for the model
        for folder in self.model_dirs[model_type]:
            file = Path(folder) / name
            if file.is_file():
                return file
        return None

    def update_all_models(self):
        for model_type in ["checkpoints", "loras", "inbox"]:
            threading.Thread(
                target=self.civit_update_worker,
                args=(
                    model_type,
                    self.model_dirs[model_type],
                ),
                daemon=True,
            ).start()

    def __init__(self, offline=False):
        self.offline = offline

        self.ready = {
            "checkpoints": False,
            "loras": False,
            "inbox": False,
        }
        self.names = {
            "checkpoints": [],
            "loras": [],
            "inbox": [],
        }
        checkpoints = path_manager.model_paths["modelfile_path"]
        checkpoints = checkpoints if isinstance(checkpoints, list) else [checkpoints]
        loras = path_manager.model_paths["lorafile_path"]
        loras = loras if isinstance(loras, list) else [loras]
        inbox = path_manager.model_paths["inbox_path"]
        inbox = inbox if isinstance(inbox, list) else [inbox]
        self.model_dirs = {
            "checkpoints": checkpoints,
            "loras": loras,
            "inbox": inbox,
        }
        self.cache_paths = {
            "checkpoints": Path(path_manager.model_paths["cache_path"] / "checkpoints"),
            "loras": Path(path_manager.model_paths["cache_path"] / "loras"),
            "inbox": Path(path_manager.model_paths["cache_path"] / "inbox"),
        }

        self.base_url = "https://civitai.com/api/v1/"
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.EXTENSIONS = [".pth", ".ckpt", ".bin", ".safetensors", ".gguf"]

        self.update_all_models()


    def get_file_from_name(self, model_type, model_name):
        for folder in self.model_dirs[model_type]:
            path = Path(folder) / model_name
            if path.is_file():
                return path
        return None

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

    def get_models_by_path(self, model_type, path):
        data = None

        cache_path = Path(self.cache_paths[model_type]) / Path(Path(path).name)
        if cache_path.is_dir():
            # Give up
            return {}
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
        from modules.settings import default_settings
        if "model_preview" in default_settings:
            opts = default_settings["model_preview"].split(",")
            if "caption" in opts:
                caption=True
            if "nogifzoom" in opts:
                nogifzoom=True
            if "zoom" in opts:
                zoom=True
        else:
            caption=False
            nogifzoom=False
            zoom=False

        def make_thumbnail(image, text, zoom=False, caption=False):
            max = 166  # Max width or height

            if image is None:
                return None

            if zoom:
                oh = image.shape[0]
                ow = image.shape[1]
                scale = max / oh if oh > ow else max / ow
                image = cv2.resize(
                    image,
                    dsize=(int(ow * scale), int(oh * scale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if caption:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.35
                thickness = 1

                org = (3, 10)
                color = (25, 15, 11) # BGR
                image = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness*2,
                    cv2.LINE_AA
                )
                org = (3, 10)
                color = (255, 215, 185) # BGR
                image = cv2.putText(
                    image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

            return image

        path = path.with_suffix(".jpeg")
        caption_text = f"{path.with_suffix('').name}"

        image_url = None
        for preview in model.get("images", [{}]):
            url = preview.get("url")
            format = preview.get("type")
            if url:
                print(f"Updating preview for {caption_text}.")
                image_url = url
                response = self.session.get(image_url)
                if response.status_code != 200:
                    print(f"WARNING: get_image() - {response.status_code} : {response.reason}")
                    break
                image = np.asarray(bytearray(response.content), dtype="uint8") 
                out = make_thumbnail(cv2.imdecode(image, cv2.IMREAD_COLOR), caption_text, caption=caption, zoom=zoom)
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
                        out = make_thumbnail(i, caption_text, caption=caption, zoom=not nogifzoom)
                        if out is None:
                            out = i
                        video_out.append(out)
                    iio.imwrite(
                        str(path.with_suffix(".gif")), video_out, fps=fps, loop=0
                    )
                    os.remove(tmp_path)
                break
        if image_url is None:
            shutil.copyfile("html/warning.jpeg", path)
