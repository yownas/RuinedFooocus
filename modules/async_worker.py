import threading
import gc
import torch
import re
from playsound import playsound
from os.path import exists
from comfy.model_management import InterruptProcessingException

buffer = []
outputs = []


def worker():
    global buffer, outputs

    import json
    import os
    import time
    import shared
    import random
    import modules.default_pipeline as pipeline
    import modules.path
    import modules.patch

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

    def handler(task):
        (
            prompt,
            negative_prompt,
            style_selction,
            performance_selction,
            aspect_ratios_selction,
            image_number,
            image_seed,
            sharpness,
            save_metadata,
            cfg,
            base_clip_skip,
            refiner_clip_skip,
            sampler_name,
            scheduler,
            custom_steps,
            custom_switch,
            base_model_name,
            refiner_model_name,
            l1,
            w1,
            l2,
            w2,
            l3,
            w3,
            l4,
            w4,
            l5,
            w5,
            gallery,
        ) = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        pipeline.clean_prompt_cond_caches()

        p_txt, n_txt = apply_style(style_selction, prompt, negative_prompt)

        if performance_selction == "Speed":
            steps = 30
            switch = 20
        elif performance_selction == "Quality":
            steps = 60
            switch = 40
        else:  # Custom
            steps = custom_steps
            switch = custom_switch

        width, height = aspect_ratios[aspect_ratios_selction]

        results = []
        seed = image_seed

        max_seed = 0xFFFFFFFFFFFFFFFF
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, max_seed)
        if seed > max_seed:
            seed = seed % max_seed

        all_steps = steps * image_number
        with open("render.txt") as f:
            lines = f.readlines()
        status = random.choice(lines)

        def callback(step, x0, x, total_steps, y):
            global status
            done_steps = i * steps + step
            if step % 10 == 0:
                status = random.choice(lines)
            outputs.append(
                [
                    "preview",
                    (
                        int(100.0 * float(done_steps) / float(all_steps)),
                        i,
                        image_number,
                        f"{status} - {step}/{total_steps}",
                        width,
                        height,
                        y,
                    ),
                ]
            )

        gallery_size = len(gallery)

        stop_batch = False
        for i in range(image_number):
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
            input_image_path = None
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
                    input_image_path,
                    start_step,
                    denoise,
                    cfg,
                    base_clip_skip,
                    refiner_clip_skip,
                    sampler_name,
                    scheduler,
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
                if save_metadata:
                    prompt = {
                        "Prompt": wildcard_text,
                        "Negative": n_txt,
                        "steps": steps,
                        "switch": switch,
                        "cfg": cfg,
                        "width": width,
                        "height": height,
                        "seed": seed,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "base_model_name": base_model_name,
                        "refiner_model_name": refiner_model_name,
                        "l1": l1,
                        "w1": w1,
                        "l2": l2,
                        "w2": w2,
                        "l3": l3,
                        "w3": w3,
                        "l4": l4,
                        "w4": w4,
                        "l5": l5,
                        "w5": w5,
                        "sharpness": sharpness,
                        "start_step": start_step,
                        "denoise": denoise,
                        "software": "RuinedFooocus",
                    }
                    metadata = PngInfo()
                    metadata.add_text("parameters", json.dumps(prompt))
                Image.fromarray(x).save(local_temp_filename, pnginfo=metadata)
                results.append(local_temp_filename)

            seed += 1
            if stop_batch:
                break

        outputs.append(["results", results])
        return

    while True:
        time.sleep(0.01)
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
