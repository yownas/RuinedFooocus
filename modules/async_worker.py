import threading
import gc
import torch
import re
from playsound import playsound
from os.path import exists

buffer = []
outputs = []

interrupt_ruined_processing = False


def worker():
    global buffer, outputs

    import json
    import os
    import time
    import shared
    import random
    import modules.default_pipeline as pipeline
    import modules.path

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from modules.sdxl_styles import apply_style, aspect_ratios
    from modules.util import generate_temp_filename

    try:
        async_gradio_app = shared.gradio_root
        flag = f"""App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}"""
        if async_gradio_app.share:
            flag += f""" or {async_gradio_app.share_url}"""
        print(flag)
    except Exception as e:
        print(e)

    def handler(t):
        loras = [(t["l1"], t["w1"]), (t["l2"], t["w2"]), (t["l3"], t["w3"]), (t["l4"], t["w4"]), (t["l5"], t["w5"])]

        pipeline.refresh_base_model(t["base_model_name"])
        pipeline.refresh_refiner_model(t["refiner_model_name"])
        pipeline.refresh_loras(loras)
        pipeline.clean_prompt_cond_caches()

        p_txt, n_txt = apply_style(t["style_selction"], t["prompt"], t["negative_prompt"])

        if t["performance_selction"] == "Speed":
            steps = 30
            switch = 20
        elif t["performance_selction"] == "Quality":
            steps = 60
            switch = 40
        else:  # Custom
            steps = t["custom_steps"]
            switch = t["custom_switch"]

        width, height = aspect_ratios[t["aspect_ratios_selction"]]

        results = []
        metadatastrings = []
        seed = t["image_seed"]

        max_seed = 0xFFFFFFFFFFFFFFFF
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, max_seed)
        if seed > max_seed:
            seed = seed % max_seed

        all_steps = steps * t["image_number"]
        with open("render.txt") as f:
            lines = f.readlines()
        status = random.choice(lines)

        class InterruptProcessingException(Exception):
            pass

        def callback(step, x0, x, total_steps, y):
            global status, interrupt_ruined_processing

            if interrupt_ruined_processing:
                interrupt_ruined_processing = False
                raise InterruptProcessingException()

            done_steps = i * steps + step
            if step % 10 == 0:
                status = random.choice(lines)
            outputs.append(
                [
                    "preview",
                    (
                        int(100.0 * float(done_steps) / float(all_steps)),
                        i,
                        t["image_number"],
                        f"{status} - {step}/{total_steps}",
                        width,
                        height,
                        y,
                    ),
                ]
            )

        stop_batch = False
        for i in range(t["image_number"]):
            directory = "wildcards"
            wildcard_text = p_txt
            placeholders = re.findall(r"__(\w+)__", wildcard_text)
            for placeholder in placeholders:
                try:
                    with open(os.path.join(directory, f"{placeholder}.txt")) as f:
                        words = f.read().splitlines()
                    wildcard_text = re.sub(rf"__{placeholder}__", random.choice(words), wildcard_text)
                except IOError:
                    print(
                        f"Error: Could not open file {placeholder}.txt. Please ensure the file exists and is readable."
                    )
                    raise
            start_step = 0
            denoise = None
            start_time = time.time()
            pipeline.clean_prompt_cond_caches()
            try:
                imgs = pipeline.process(
                    wildcard_text,
                    n_txt,
                    steps,
                    switch,
                    width,
                    height,
                    seed,
                    start_step,
                    denoise,
                    t["cfg"],
                    t["base_clip_skip"],
                    t["refiner_clip_skip"],
                    t["sampler_name"],
                    t["scheduler"],
                    callback=callback,
                )
            except InterruptProcessingException as iex:
                stop_batch = True
                imgs = []
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\033[91mTime taken: {elapsed_time:0.2f} seconds\033[0m")

            for x in imgs:
                local_temp_filename = generate_temp_filename(folder=modules.path.temp_outputs_path, extension="png")
                os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)
                metadata = None
                if t["save_metadata"]:
                    prompt = {
                        "Prompt": wildcard_text,
                        "Negative": n_txt,
                        "steps": steps,
                        "switch": switch,
                        "cfg": t["cfg"],
                        "width": width,
                        "height": height,
                        "seed": seed,
                        "sampler_name": t["sampler_name"],
                        "scheduler": t["scheduler"],
                        "base_model_name": t["base_model_name"],
                        "refiner_model_name": t["refiner_model_name"],
                        "loras": "Loras:" + ",".join([f"<{lora[0]}:{lora[1]}>" for lora in loras]),
                        "start_step": start_step,
                        "denoise": denoise,
                        "software": "RuinedFooocus",
                    }
                    metadata = PngInfo()
                    metadata.add_text("parameters", json.dumps(prompt))
                Image.fromarray(x).save(local_temp_filename, pnginfo=metadata)
                results.append(local_temp_filename)
                metadatastrings.append(prompt)

            seed += 1
            if stop_batch:
                break

        outputs.append(["results", results])
        outputs.append(["metadata", metadatastrings])
        return

    while True:
        time.sleep(0.1)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if exists("notification.mp3"):
                playsound("notification.mp3")
    pass


threading.Thread(target=worker, daemon=True).start()
