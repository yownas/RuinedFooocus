import argparse
import random
import shared
import time

import gradio as gr

import fooocus_version
import modules.async_worker as worker
import modules.html
import modules.path
import ui_onebutton

from comfy.samplers import KSampler
from modules.sdxl_styles import style_keys, aspect_ratios, styles
from modules.settings import default_settings
from modules.prompt_processing import get_promptlist


import math
from PIL import Image


state = {"preview_image": None, "ctrls_name": [], "ctrls_obj": []}


def load_images_handler(files):
    return list(map(lambda x: x.name, files))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
    parser.add_argument("--share", action="store_true", help="Set whether to share on Gradio.")
    parser.add_argument(
        "--listen",
        type=str,
        default=None,
        metavar="IP",
        nargs="?",
        const="0.0.0.0",
        help="Set the listen interface.",
    )
    parser.add_argument("--nobrowser", action="store_true", help="Do not launch in browser.")
    return parser


def parse_args():
    parser = get_parser()
    return parser.parse_args()


def launch_app(args):
    inbrowser = not args.nobrowser
    favicon_path = "logo.ico"
    shared.gradio_root.queue(concurrency_count=4)
    shared.gradio_root.launch(
        inbrowser=inbrowser,
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        favicon_path=favicon_path,
    )


def generate_preview(image_nr, image_cnt, width, height, image):
    grid_xsize = math.ceil(math.sqrt(image_cnt))
    grid_ysize = math.ceil(image_cnt / grid_xsize)
    grid_max = max(grid_xsize, grid_ysize)
    pwidth = int(width * grid_xsize / grid_max)
    pheight = int(height * grid_ysize / grid_max)
    if state["preview_image"] is None:
        state["preview_image"] = Image.new("RGB", (pwidth, pheight))
    if image is not None:
        image = Image.fromarray(image)
        grid_xpos = int((image_nr % grid_xsize) * (pwidth / grid_xsize))
        grid_ypos = int(math.floor(image_nr / grid_xsize) * (pheight / grid_ysize))
        image = image.resize((int(width / grid_max), int(height / grid_max)))
        state["preview_image"].paste(image, (grid_xpos, grid_ypos))


def update_clicked():
    return {
        run_button: gr.update(interactive=False, visible=False),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True,
            value=modules.html.make_progress_html(0, "Processing text encoding ..."),
        ),
        progress_window: gr.update(visible=True, value=None),
        metadata_viewer: gr.update(),
        gallery: gr.update(visible=False),
    }


def update_preview(percentage, title, preview_image_path):
    return {
        run_button: gr.update(interactive=False, visible=False),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)),
        progress_window: gr.update(visible=True, value=preview_image_path)
        if preview_image_path is not None
        else gr.update(),
        metadata_viewer: gr.update(visible=False),
        gallery: gr.update(visible=False),
    }


def update_results(product):
    return {
        run_button: gr.update(interactive=True, visible=True),
        stop_button: gr.update(interactive=False, visible=False),
        progress_html: gr.update(visible=False),
        progress_window: gr.update(),
        metadata_viewer: gr.update(),
        gallery: gr.update(visible=True, value=product),
    }


def update_metadata(product):
    return {
        run_button: gr.update(),
        stop_button: gr.update(),
        progress_html: gr.update(),
        progress_window: gr.update(),
        metadata_viewer: gr.update(value=product),
        gallery: gr.update(),
    }


def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]


def generate_clicked(*args):
    state["preview_image"] = None
    yield update_clicked()
    gen_data = {}
    for key, val in zip(state["ctrls_name"], args):
        gen_data[key] = val

    prompts = get_promptlist(gen_data)
    idx = 0
    for prompt in prompts:
        gen_data["prompt"] = prompt
        gen_data["index"] = (idx, len(prompts))
        idx += 1
        worker.buffer.append(gen_data.copy())

    finished = False
    while not finished:
        time.sleep(0.1)

        if not worker.outputs:
            continue

        flag, product = worker.outputs.pop(0)

        if flag == "preview":
            percentage, image_nr, image_cnt, title, width, height, image = product
            generate_preview(image_nr, image_cnt, width, height, image)
            preview_image_path = "outputs/preview.jpg"
            state["preview_image"].save(preview_image_path, optimize=True, quality=35)
            yield update_preview(percentage, title, preview_image_path)

        elif flag == "results":
            yield update_results(product)

        elif flag == "metadata":
            yield update_metadata(product)
            finished = True

    state["preview_image"] = None


settings = default_settings

if settings["theme"] == "None":
    theme = gr.themes.Default()
else:
    theme = settings["theme"]

shared.gradio_root = gr.Blocks(
    theme=theme, title="RuinedFooocus " + fooocus_version.version, css=modules.html.css
).queue()
with shared.gradio_root as block:
    block.load(_js=modules.html.scripts)
    with gr.Row():
        with gr.Column(scale=5):
            progress_window = gr.Image(
                value="init_image.png", height=680, type="pil", visible=True, show_label=False, image_mode="RGBA"
            )
            progress_html = gr.HTML(
                value=modules.html.make_progress_html(32, "Progress 32%"),
                visible=False,
                elem_id="progress-bar",
                elem_classes="progress-bar",
            )

            gallery = gr.Gallery(
                label="Gallery",
                show_label=False,
                object_fit="scale-down",
                height=60,
                allow_preview=True,
                preview=True,
                visible=True,
            )

            def gallery_change(files, sd: gr.SelectData):
                return files[sd.index]["name"]

            gallery.select(gallery_change, [gallery], outputs=[progress_window], show_progress="hidden")

            with gr.Row(elem_classes="type_row"):
                with gr.Column(scale=5):
                    prompt = gr.Textbox(
                        show_label=False,
                        placeholder="Type prompt here.",
                        container=False,
                        autofocus=True,
                        elem_classes="type_row",
                        lines=1024,
                        value=settings["prompt"],
                    )
                    add_ctrl("prompt", prompt)

                with gr.Column(scale=1, min_width=0):
                    run_button = gr.Button(label="Generate", value="Generate", elem_id="generate")
                    stop_button = gr.Button(label="Stop", value="Stop", interactive=False, visible=False)

                    def load_images_handler(file):
                        info = file.info
                        params = info.get("parameters", "")
                        return params, [file]

                    progress_window.upload(load_images_handler, inputs=[progress_window], outputs=[prompt, gallery])

            with gr.Row():
                advanced_checkbox = gr.Checkbox(
                    label="Hurt me plenty", value=settings["advanced_mode"], container=False
                )
        with gr.Column(scale=2, visible=settings["advanced_mode"]) as right_col:
            with gr.Tab(label="Setting"):
                performance_selection = gr.Radio(
                    label="Performance",
                    choices=["Speed", "Quality", "Custom"],
                    value=settings["performance"],
                )
                add_ctrl("performance_selection", performance_selection)
                aspect_ratios_selection = gr.Dropdown(
                    label="Aspect Ratios (width x height)",
                    choices=list(aspect_ratios.keys()),
                    value=settings["resolution"],
                )
                add_ctrl("aspect_ratios_selection", aspect_ratios_selection)
                style_selection = gr.Dropdown(
                    label="Style Selection",
                    multiselect=True,
                    container=True,
                    choices=style_keys,
                    value=settings["style"],
                )
                add_ctrl("style_selection", style_selection)
                style_button = gr.Button(value="⬅️ Send Style to prompt", size="sm")
                image_number = gr.Slider(
                    label="Image Number",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=settings["image_number"],
                )
                add_ctrl("image_number", image_number)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    show_label=True,
                    placeholder="Type prompt here.",
                    value=settings["negative_prompt"],
                )
                add_ctrl("negative", negative_prompt)
                seed_random = gr.Checkbox(label="Random Seed", value=settings["seed_random"])
                image_seed = gr.Number(
                    label="Seed",
                    value=settings["seed"],
                    precision=0,
                    visible=not settings["seed_random"],
                )
                add_ctrl("seed", image_seed)

                def apply_style(prompt_test, inputs):
                    pr = ""
                    ne = ""
                    for item in inputs:
                        p, n = styles.get(item)
                        pr += p + ", "
                        ne += n + ", "
                    if prompt_test:
                        pr = pr.replace("{prompt}", prompt_test)
                    return pr, ne, []

                style_button.click(
                    apply_style,
                    inputs=[prompt, style_selection],
                    outputs=[prompt, negative_prompt, style_selection],
                )

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, s):
                    if r:
                        return random.randint(1, 1024 * 1024 * 1024)
                    else:
                        return s

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed])

            with gr.Tab(label="Models"):
                with gr.Row():
                    base_model = gr.Dropdown(
                        label="SDXL Base Model",
                        choices=modules.path.model_filenames,
                        value=settings["base_model"],
                        show_label=True,
                    )
                    add_ctrl("base_model_name", base_model)
                    refiner_model = gr.Dropdown(
                        label="SDXL Refiner",
                        choices=["None"] + modules.path.model_filenames,
                        value=settings["refiner_model"],
                        show_label=True,
                    )
                    add_ctrl("refiner_model_name", refiner_model)
                with gr.Accordion(label="LoRAs", open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(
                                label=f"SDXL LoRA {i+1}",
                                choices=["None"] + modules.path.lora_filenames,
                                value=settings[f"lora_{i+1}_model"],
                            )
                            add_ctrl(f"l{i+1}", lora_model)

                            lora_weight = gr.Slider(
                                label="Strength",
                                minimum=-2,
                                maximum=2,
                                step=0.05,
                                value=settings[f"lora_{i+1}_weight"],
                            )
                            add_ctrl(f"w{i+1}", lora_weight)
                            lora_ctrls += [lora_model, lora_weight]
                with gr.Row():
                    model_refresh = gr.Button(
                        label="Refresh",
                        value="\U0001f504 Refresh All Files",
                        variant="secondary",
                        elem_classes="refresh_button",
                    )
            with gr.Tab(label="Advanced"):
                save_metadata = gr.Checkbox(label="Save Metadata", value=settings["save_metadata"])
                add_ctrl("save_metadata", save_metadata)
                custom_steps = gr.Slider(
                    label="Custom Steps",
                    minimum=10,
                    maximum=200,
                    step=1,
                    value=30,
                    visible=False,
                )
                add_ctrl("custom_steps", custom_steps)
                custom_switch = gr.Slider(
                    label="Custom Switch",
                    minimum=10,
                    maximum=200,
                    step=1,
                    value=20,
                    visible=False,
                )
                add_ctrl("custom_switch", custom_switch)

                cfg = gr.Slider(
                    label="CFG",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    value=8,
                    visible=False,
                )
                add_ctrl("cfg", cfg)
                base_clip_skip = gr.Slider(
                    label="Base CLIP Skip",
                    minimum=-10,
                    maximum=-1,
                    step=1,
                    value=-2,
                    visible=False,
                )
                add_ctrl("base_clip_skip", base_clip_skip)
                refiner_clip_skip = gr.Slider(
                    label="Refiner CLIP Skip",
                    minimum=-10,
                    maximum=-1,
                    step=1,
                    value=-2,
                    visible=False,
                )
                add_ctrl("refiner_clip_skip", refiner_clip_skip)
                sampler_name = gr.Dropdown(
                    label="Sampler",
                    choices=KSampler.SAMPLERS,
                    value="dpmpp_2m_sde_gpu",
                    visible=False,
                )
                add_ctrl("sampler_name", sampler_name)
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=KSampler.SCHEDULERS,
                    value="karras",
                    visible=False,
                )
                add_ctrl("scheduler", scheduler)

                performance_outputs = [
                    cfg,
                    base_clip_skip,
                    refiner_clip_skip,
                    sampler_name,
                    scheduler,
                    custom_steps,
                    custom_switch,
                ]

                def performance_changed(selection):
                    if selection != "Custom":
                        return [gr.update(visible=False)] * len(performance_outputs)
                    else:
                        return [gr.update(visible=True)] * len(performance_outputs)

                performance_selection.change(
                    performance_changed,
                    inputs=[performance_selection],
                    outputs=performance_outputs,
                )

                metadata_viewer = gr.JSON(label="Metadata", elem_classes="json-container")

            def model_refresh_clicked():
                modules.path.update_all_model_names()
                results = []
                results += [
                    gr.update(choices=modules.path.model_filenames),
                    gr.update(choices=["None"] + modules.path.model_filenames),
                ]
                for i in range(5):
                    results += [
                        gr.update(choices=["None"] + modules.path.lora_filenames),
                        gr.update(),
                    ]
                return results

            model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls)

            ui_onebutton.ui_onebutton(prompt)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, right_col)

        run_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed).then(
            fn=generate_clicked,
            inputs=state["ctrls_obj"],
            outputs=[run_button, stop_button, progress_html, progress_window, metadata_viewer, gallery],
        )

        def stop_clicked():
            worker.buffer = []
            worker.interrupt_ruined_processing = True

        stop_button.click(fn=stop_clicked, queue=False)

args = parse_args()
launch_app(args)
