import gradio as gr
import os
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


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


class ImageBrowser:
    def __init__(self):
        self.images = []
        self.base_path = None
        self.current_display_paths = []  # Track currently displayed images

    def load_images(self, folder_path: str) -> Tuple[List[str], str]:
        """Load all PNG images and return paths and status message."""
        if not folder_path:
            return [], "Please enter a folder path"

        try:
            self.base_path = Path(folder_path)
            if not self.base_path.exists():
                return [], f"Folder not found: {folder_path}"

            image_paths = []
            self.images = []  # Reset images list

            # Walk through directory and all subdirectories
            for root, _, files in os.walk(self.base_path):
                for filename in files:
                    if filename.lower().endswith(".png"):
                        full_path = Path(root) / filename
                        rel_path = str(full_path.relative_to(self.base_path))
                        metadata = get_png_metadata(str(full_path))
                        metadata["file_path"] = rel_path

                        image_paths.append(str(full_path))
                        self.images.append((str(full_path), metadata))

            self.current_display_paths = image_paths  # Store current display order

            if image_paths:
                return (
                    image_paths,
                    f"Loaded {len(image_paths)} images from {folder_path} and its subdirectories",
                )
            return [], f"No PNG images found in {folder_path} or its subdirectories"

        except Exception as e:
            return [], f"Error loading folder: {e}"

    def get_image_metadata(self, evt: gr.SelectData) -> str:
        """Get metadata for selected image."""
        try:
            selected_path = self.current_display_paths[evt.index]
            matching_metadata = next(
                metadata for path, metadata in self.images if path == selected_path
            )
            return format_metadata_string(matching_metadata)
        except Exception as e:
            return f"Error getting metadata: {e}"

    def search_metadata(self, search_term: str) -> Tuple[List[str], str]:
        """Filter images based on metadata search."""
        if not search_term:
            self.current_display_paths = [path for path, _ in self.images]
            return self.current_display_paths, "Showing all images"

        search_term = search_term.lower()
        matching_paths = []

        for image_path, metadata in self.images:
            # Search in formatted metadata
            formatted = format_metadata(metadata)
            metadata_str = json.dumps(formatted).lower()
            if search_term in metadata_str:
                matching_paths.append(image_path)

        self.current_display_paths = (
            matching_paths if matching_paths else [path for path, _ in self.images]
        )

        if matching_paths:
            return matching_paths, f"Found {len(matching_paths)} matching images"
        return self.current_display_paths, "No matches found - showing all images"
