import threading
import gc
import torch
import math
from playsound import playsound
from os.path import exists

buffer = []
outputs = []
results = []
metadatastrings = []

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
    from modules.prompt_processing import process_metadata, process_prompt, parse_loras

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
        match gen_data["task_type"]:
            case "start":
                job_start(gen_data)
            case "stop":
                job_stop()
            case "process":
                process(gen_data)
            case _:
                print(f"WARN: Unknown task_type: {gen_data['task_type']}")

    def job_start(gen_data):
        from shared import state
        state["preview_image"] = None
        state["preview_count"] = gen_data["image_count"]
        state["preview_current"] = 0

    def job_stop():
        from shared import state
        state["preview_image"] = None
        state["preview_count"] = 0
        state["preview_current"] = 0

    def process(gen_data):
        global results, metadatastrings
        from shared import state

        gen_data = process_metadata(gen_data)

        loras = []
        i = 1

        while True:
            l_key = f"l{i}"
            w_key = f"w{i}"
            try:
                loras.append((gen_data[l_key], gen_data[w_key]))
                i += 1
            except KeyError:
                break

        parsed_loras, pos_stripped, neg_stripped = parse_loras(gen_data["prompt"], gen_data["negative"])
        loras.extend(parsed_loras)

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
        if "width" in gen_data:
            width = gen_data["width"]
        if "height" in gen_data:
            height = gen_data["height"]

        seed = gen_data["seed"]

        max_seed = 0xFFFFFFFFFFFFFFFF
        if not isinstance(seed, int) or seed < 0:
            seed = random.randint(0, max_seed)
        seed = seed % max_seed

        all_steps = steps * gen_data["image_number"]
        with open("render.txt") as f:
            lines = f.readlines()
        status = random.choice(lines)

        class InterruptProcessingException(Exception):
            pass

        def callback(step, x0, x, total_steps, y):
            global status, interrupt_ruined_processing
            from shared import state

            if interrupt_ruined_processing:
                interrupt_ruined_processing = False
                raise InterruptProcessingException()

            done_steps = i * steps + step
            if step % 10 == 0:
                status = random.choice(lines)

            grid_xsize = math.ceil(math.sqrt(state["preview_count"]))
            grid_ysize = math.ceil(state["preview_count"] / grid_xsize)
            grid_max = max(grid_xsize, grid_ysize)
            pwidth = int(width * grid_xsize / grid_max)
            pheight = int(height * grid_ysize / grid_max)
            if state["preview_image"] is None:
                state["preview_image"] = Image.new("RGB", (pwidth, pheight))
            if y is not None:
                image = Image.fromarray(y)
                grid_xpos = int((state["preview_current"] % grid_xsize) * (pwidth / grid_xsize))
                grid_ypos = int(math.floor(state["preview_current"] / grid_xsize) * (pheight / grid_ysize))
                image = image.resize((int(width / grid_max), int(height / grid_max)))
                state["preview_image"].paste(image, (grid_xpos, grid_ypos))

            preview_image_path = "outputs/preview.jpg" # FIXME move somewhere else
            state["preview_image"].save(preview_image_path, optimize=True, quality=35)

            outputs.append(
                [
                    "preview",
                    (
                        int(100 * (gen_data["index"][0] + done_steps / all_steps) / gen_data["index"][1]),
                        f"{status} - {step}/{total_steps}",
                        preview_image_path,
                    ),
                ]
            )

        stop_batch = False
        for i in range(gen_data["image_number"]):
            p_txt, n_txt = process_prompt(gen_data["style_selection"], pos_stripped, neg_stripped)
            start_step = 0
            denoise = None
            start_time = time.time()
            pipeline.clean_prompt_cond_caches()
            try:
                imgs = pipeline.process(
                    p_txt,
                    n_txt,
                    gen_data["input_image"],
                    gen_data["controlnet_selection"],
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

                state["preview_current"] += 1
                Image.fromarray(x).save(local_temp_filename, pnginfo=metadata)
                results.append(local_temp_filename)
                metadatastrings.append(json.dumps(prompt))

            seed += 1
            if stop_batch:
                break

        if len(buffer) == 0:
            if state["preview_image"] is not None and state["preview_count"] > 1:
                results = [state["preview_image"]] + results
            outputs.append(["results", results])
            outputs.append(["metadata", metadatastrings])
            results = []
            metadatastrings = []
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
