import gradio as gr
import os
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sqlite3
import time
from modules.path import PathManager
from modules.util import TimeIt
from modules.settings import default_settings
import version


def format_metadata(metadata: Dict) -> Dict:
    """Format metadata into a more readable structure."""
    try:
        # Create formatted output dictionary
        formatted = {"File Path": metadata.get("file_path", "Unknown")}

        # Parse the parameters string if it exists
        if "parameters" in metadata:
            params = json.loads(metadata["parameters"])

            # Add generation parameters in a logical order
            if "Prompt" in params:
                formatted["Prompt"] = params["Prompt"].strip()
            if "Negative" in params:
                formatted["Negative Prompt"] = params["Negative"].strip()

            # Model information
            if "base_model_name" in params:
                formatted["Model"] = params["base_model_name"]
            if "base_model_hash" in params:
                formatted["Model Hash"] = params["base_model_hash"]

            # Generation settings
            settings = {}
            for key in [
                "steps",
                "cfg",
                "width",
                "height",
                "seed",
                "sampler_name",
                "scheduler",
                "clip_skip",
                "denoise",
            ]:
                if key in params and params[key] is not None:
                    settings[key.capitalize()] = params[key]
            formatted["Settings"] = settings

            # Software info
            if "software" in params:
                formatted["Software"] = params["software"]

            # LoRAs if present
            if "loras" in params and params["loras"]:
                formatted["LoRAs"] = params["loras"]

        return formatted
    except Exception as e:
        print(f"Error formatting metadata: {e}")
        return metadata


def format_metadata_string(metadata: Dict) -> str:
    """Convert formatted metadata into a readable string."""
    try:
        formatted = format_metadata(metadata)
        output = []

        # File path
        output.append(f"File: {formatted['File Path']}\n")

        # Prompt
        if "Prompt" in formatted:
            output.append("Prompt:")
            output.append(formatted["Prompt"])
            output.append("")

        # Negative prompt
        if "Negative Prompt" in formatted and formatted["Negative Prompt"]:
            output.append("Negative Prompt:")
            output.append(formatted["Negative Prompt"])
            output.append("")

        # Model info
        if "Model" in formatted:
            output.append(f"Model: {formatted['Model']}")
            if "Model Hash" in formatted:
                output.append(f"Hash: {formatted['Model Hash']}")
            output.append("")

        # Settings
        if "Settings" in formatted:
            output.append("Generation Settings:")
            for key, value in formatted["Settings"].items():
                output.append(f"  {key}: {value}")
            output.append("")

        # Software
        if "Software" in formatted:
            output.append(f"Software: {formatted['Software']}")

        # LoRAs
        if "LoRAs" in formatted and formatted["LoRAs"]:
            output.append("\nLoRAs:")
            for lora in formatted["LoRAs"]:
                output.append(f"  {lora}")

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting metadata string: {e}\n\nRaw metadata:\n{json.dumps(metadata, indent=2)}"


def get_png_metadata(image_path: str) -> Dict:
    """Extract metadata from PNG file."""
    try:
        with Image.open(image_path) as img:
            if isinstance(img, PngImageFile):
                metadata = {}
                for key, value in img.info.items():
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8")
                        except UnicodeDecodeError:
                            value = str(value)
                    metadata[key] = value
                return metadata
            return {}
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
        return {}


def connect_database(path="cache/images.db"):
    # Connect to an SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(path, check_same_thread=False)

    conn.cursor()
    conn.execute('''CREATE TABLE IF NOT EXISTS status (version text, date text)''')
    res = conn.execute("SELECT count(*) FROM status")
    cnt = res.fetchone()[0]
    newdb = False
    if cnt == 0:
        conn.execute(
            "INSERT INTO status(version, date) VALUES (?,?)",
            (str(version.version), str(time.time()))
        )
        newdb = True
    else:
        res = conn.execute("SELECT version FROM status")
        dbver = res.fetchone()[0]
        if dbver != version.version:
            newdb = True

    if newdb:
        try:
            conn.execute("DROP TABLE images")
        except:
            pass
    conn.commit()
    conn.cursor()
    conn.execute('''CREATE TABLE IF NOT EXISTS images (fullpath text, path text, json text)''')
    conn.commit()

    return conn


class ImageBrowser:
    def __init__(self):
        self.path_manager = PathManager()
        self.base_path = Path(self.path_manager.model_paths["temp_outputs_path"])
        self.current_display_paths = []  # Track currently displayed images
        self.sql_conn = connect_database()
        self.images_per_page = default_settings.get("images_per_page", 100)
        self.filter = ""

    def num_images_pages(self):
        result = self.sql_conn.execute(f"SELECT count(*) FROM images WHERE json LIKE '%{self.filter}%'") # FIXME!!! should only match prompt?
        image_cnt = result.fetchone()[0]
        pages = int(image_cnt/self.images_per_page) + 1
        return image_cnt, pages

    def load_images(self, page: int) -> Tuple[List[str], str]:
        text = ""
        if page == None:
            page = 1
        result = self.sql_conn.execute(
            f"SELECT fullpath, path FROM images WHERE json LIKE '%{self.filter}%' ORDER BY path DESC LIMIT ? OFFSET ?",
            (
                str(self.images_per_page),
                str((page-1)*self.images_per_page),
            )
        )
        image_paths = result.fetchall()
        self.current_display_paths = image_paths  # Store current display order
        if image_paths:
            path1 = str(Path(image_paths[0][1]))
            path2 = str(Path(image_paths[-1][1]))
        else:
            path1 = "None"
            path2 = "None"
        text = f"{path1} ... {path2}"

        if image_paths:
            return list(Path(x[0]) for x in image_paths), text
        return [], text

    def update_images(self) -> Tuple[List[str], str]:
        """Check all images and update database"""
        #try:
        if True:
            if not self.base_path.exists():
                return [], f"Folder not found: {self.base_path}"

            image_cnt = 0
            self.sql_conn.cursor()
            self.sql_conn.execute("DROP TABLE images")
            self.sql_conn.commit()
            self.sql_conn = connect_database()
            self.sql_conn.cursor()

            # Walk through directory and all subdirectories
            with TimeIt("Update DB"):
                for folder in [self.base_path] + default_settings.get("archive_folders", []):
                    print(f"DEBUG: {folder}")
                    for root, _, files in os.walk(folder):
                        print(f"DEBUG: {files}")
                        for filename in files:
                            #if filename.lower().endswith((".png", ".gif")):
                            if filename.lower().endswith(".png"):
                                full_path = Path(root) / filename
                                rel_path = str(full_path.relative_to(folder))
                                if filename.lower().endswith(".png"):
                                    metadata = get_png_metadata(str(full_path))
                                else:
                                    metadata = {} # FIXME fake data for non-png images
                                metadata["file_path"] = rel_path

                                self.sql_conn.execute(
                                    "INSERT INTO images(fullpath, path, json) VALUES (?, ?,?)",
                                    (str(full_path), str(rel_path), json.dumps(metadata))
                                )
                                image_cnt += 1

            self.sql_conn.commit()

            if image_cnt:
                return (
                    gr.update(value=self.load_images(1)[0]),
                    gr.update(
                        value=1,
                        maximum=int(image_cnt/self.images_per_page) + 1,
                    ),
                    gr.update(
                        value=f"Found {image_cnt} images from {self.base_path} and its subdirectories",
                    )
                )
            return (
                gr.update(value=[]),
                gr.update(value=1, maximum=1),
                gr.update(value=f"No images found in {self.base_path} or its subdirectories")
            )

        #except Exception as e:
        #    return (
        #        gr.update(value=["html/error.png"]),
        #        gr.update(value=1, maximum=1),
        #        gr.update(value=f"Error updating folder: {e}")
        #    )

    def get_image_metadata(self, evt: gr.SelectData) -> str:
        """Get metadata for selected image."""
        try:
            selected_path = self.current_display_paths[evt.index][0]
            result = self.sql_conn.execute("SELECT json FROM images WHERE fullpath = ?", (str(selected_path),))
            data = json.loads(result.fetchone()[0])
            return format_metadata_string(data)
        except Exception as e:
            return f"Error getting metadata: {e}"

    def search_metadata(self, search_term: str) -> Tuple[List[str], str]:
        self.filter = search_term
        images = self.load_images(1)[0]
        num_images, num_pages = self.num_images_pages()
        text = f"Found {num_images} images"
        return (
            gr.update(value=images),
            gr.update(value=1, maximum=num_pages),
            gr.update(value=text)
        )