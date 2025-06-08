import modules.async_worker as worker
from PIL import Image

# Copy this file, add suitable code and add logic to modules/pipelines.py to select it


class pipeline:
    pipeline_type = ["template"]

    model_hash = ""

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["ruinedfooocus_was_here"] = True
        return gen_data

    def load_base_model(self, name, hash=None):
        # Check if model is already loaded
        if self.model_hash == name:
            return
        print(f"Loading model: {name}")
        self.model_hash = name
        return

    def load_keywords(self, lora):
        filename = lora.replace(".safetensors", ".txt")
        try:
            with open(filename, "r") as file:
                data = file.read()
            return data
        except FileNotFoundError:
            return " "

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Generating ...", None)
        )

        image = Image.open("html/logo.png")

        # Return finished image to preview
        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], image)

        return [image]
