import threading
import gc
import torch
from playsound import playsound
from os.path import exists
import re

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
    from modules.prompt_processing import process_prompt

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from modules.sdxl_styles import aspect_ratios
    from modules.util import generate_temp_filename

    try:
        async_gradio_app = shared.gradio_root
        flag = f"""App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}"""
        if async_gradio_app.share:
            flag += f""" or {async_gradio_app.share_url}"""
        print(flag)
    except Exception as e:
        print(e)

    def handler(gen_data):
        try:
            meta = json.loads(gen_data["prompt"])
            meta = dict((k.lower(), v) for k, v in meta.items())
            gen_data.update(meta)
            if "prompt" in meta:
                gen_data["style_selection"] = None
            if "loras" in meta:
                idx = 1
                for lora in re.findall(r"<(.*?):(.*?)>", meta["loras"]):
                    l, w = lora
                    gen_data[f"l{idx}"] = l
                    gen_data[f"w{idx}"] = float(w)
                    idx += 1
        except ValueError as e:
            pass

        loras = [
            (gen_data["l1"], gen_data["w1"]),
            (gen_data["l2"], gen_data["w2"]),
            (gen_data["l3"], gen_data["w3"]),
            (gen_data["l4"], gen_data["w4"]),
            (gen_data["l5"], gen_data["w5"]),
        ]

        pipeline.load_base_model(gen_data["base_model_name"])
        pipeline.load_refiner_model(gen_data["refiner_model_name"])
        pipeline.load_loras(loras)
        pipeline.clean_prompt_cond_caches()

        if gen_data["performance_selection"] == "Speed":
            steps = 30
            switch = 20
        elif gen_data["performance_selection"] == "Quality":
            steps = 60
            switch = 40
        else:  # Custom
            steps = gen_data["custom_steps"]
            switch = gen_data["custom_switch"]

        width, height = aspect_ratios[gen_data["aspect_ratios_selection"]]

        results = []
        metadatastrings = []
        seed = gen_data["seed"]

        max_seed = 0xFFFFFFFFFFFFFFFF
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, max_seed)
        if seed > max_seed:
            seed = seed % max_seed

        all_steps = steps * gen_data["image_number"]
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
                        gen_data["image_number"],
                        f"{status} - {step}/{total_steps}",
                        width,
                        height,
                        y,
                    ),
                ]
            )

        stop_batch = False
        for i in range(gen_data["image_number"]):
            p_txt, n_txt = process_prompt(gen_data["style_selection"], gen_data["prompt"], gen_data["negative"])
            start_step = 0
            denoise = None
            start_time = time.time()
            pipeline.clean_prompt_cond_caches()
            try:
                imgs = pipeline.process(
                    p_txt,
                    n_txt,
                    steps,
                    switch,
                    width,
                    height,
                    seed,
                    start_step,
                    denoise,
                    gen_data["cfg"],
                    gen_data["base_clip_skip"],
                    gen_data["refiner_clip_skip"],
                    gen_data["sampler_name"],
                    gen_data["scheduler"],
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
                prompt = {
                    "Prompt": p_txt,
                    "Negative": n_txt,
                    "steps": steps,
                    "switch": switch,
                    "cfg": gen_data["cfg"],
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "sampler_name": gen_data["sampler_name"],
                    "scheduler": gen_data["scheduler"],
                    "base_model_name": gen_data["base_model_name"],
                    "refiner_model_name": gen_data["refiner_model_name"],
                    "loras": "Loras:" + ",".join([f"<{lora[0]}:{lora[1]}>" for lora in loras]),
                    "start_step": start_step,
                    "denoise": denoise,
                    "software": "RuinedFooocus",
                }
                if gen_data["save_metadata"]:
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
