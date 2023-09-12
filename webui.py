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

import math
from PIL import Image


preview_image = None


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
    global preview_image
    grid_xsize = math.ceil(math.sqrt(image_cnt))
    grid_ysize = math.ceil(image_cnt / grid_xsize)
    grid_max = max(grid_xsize, grid_ysize)
    pwidth = int(width * grid_xsize / grid_max)
    pheight = int(height * grid_ysize / grid_max)
    if preview_image is None:
        preview_image = Image.new("RGB", (pwidth, pheight))
    if image is not None:
        image = Image.fromarray(image)
        grid_xpos = int((image_nr % grid_xsize) * (pwidth / grid_xsize))
        grid_ypos = int(math.floor(image_nr / grid_xsize) * (pheight / grid_ysize))
        image = image.resize((int(width / grid_max), int(height / grid_max)))
        preview_image.paste(image, (grid_xpos, grid_ypos))


def update_clicked():
    # run_button, stop_button, progress_html, progress_window, metadata_viewer, gallery
    return (
        gr.update(interactive=False, visible=False),
        gr.update(interactive=True, visible=True),
        gr.update(
            visible=True,
            value=modules.html.make_progress_html(0, "Processing text encoding ..."),
        ),
        gr.update(visible=True, value=None),
        gr.update(),
        gr.update(visible=False),
    )


def update_preview(percentage, title, preview_image_path):
    # run_button, stop_button, progress_html, progress_window, metadata_viewer, gallery
    return (
        gr.update(interactive=False, visible=False),
        gr.update(interactive=True, visible=True),
        gr.update(
            visible=True,
            value=modules.html.make_progress_html(percentage, title),
        ),
        gr.update(visible=True, value=preview_image_path) if preview_image_path is not None else gr.update(),
        gr.update(),
        gr.update(visible=False),
    )


def update_results(product):
    # run_button, stop_button, progress_html, progress_window, metadata_viewer, gallery
    return (
        gr.update(interactive=True, visible=True),
        gr.update(interactive=False, visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(),
        gr.update(visible=True, value=product),
    )


def update_metadata(product):
    # run_button, stop_button, progress_html, progress_window, metadata_viewer, gallery
    return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=product), gr.update())


def generate_clicked(*args):
    global preview_image
    preview_image = None
    yield update_clicked()
    gen_data = {}
    (
        gen_data["prompt"],
        gen_data["negative"],
        gen_data["style_selection"],
        gen_data["performance_selection"],
        gen_data["aspect_ratios_selection"],
        gen_data["image_number"],
        gen_data["seed"],
        gen_data["save_metadata"],
        gen_data["cfg"],
        gen_data["base_clip_skip"],
        gen_data["refiner_clip_skip"],
        gen_data["sampler_name"],
        gen_data["scheduler"],
        gen_data["custom_steps"],
        gen_data["custom_switch"],
        gen_data["base_model_name"],
        gen_data["refiner_model_name"],
        gen_data["l1"],
        gen_data["w1"],
        gen_data["l2"],
        gen_data["w2"],
        gen_data["l3"],
        gen_data["w3"],
        gen_data["l4"],
        gen_data["w4"],
        gen_data["l5"],
        gen_data["w5"],
    ) = list(args)
    worker.buffer.append(gen_data)

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
            preview_image.save(preview_image_path, optimize=True, quality=35)
            yield update_preview(percentage, title, preview_image_path)

        elif flag == "results":
            yield update_results(product)

        elif flag == "metadata":
            yield update_metadata(product)
            finished = True

    preview_image = None


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
            progress_window = gr.Image(label="Preview", show_label=True, height=640, visible=False)
            progress_html = gr.HTML(
                value=modules.html.make_progress_html(32, "Progress 32%"),
                visible=False,
                elem_id="progress-bar",
                elem_classes="progress-bar",
            )
            gallery = gr.Gallery(
                label="Gallery",
                show_label=False,
                object_fit="contain",
                height=720,
                visible=True,
            )
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
                with gr.Column(scale=1, min_width=0):
                    run_button = gr.Button(label="Generate", value="Generate", elem_id="generate")
                    stop_button = gr.Button(label="Stop", value="Stop", interactive=False, visible=False)
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
                aspect_ratios_selection = gr.Dropdown(
                    label="Aspect Ratios (width x height)",
                    choices=list(aspect_ratios.keys()),
                    value=settings["resolution"],
                )
                style_selection = gr.Dropdown(
                    label="Style Selection",
                    multiselect=True,
                    container=True,
                    choices=style_keys,
                    value=settings["style"],
                )
                style_button = gr.Button(value="⬅️ Send Style to prompt", size="sm")
                image_number = gr.Slider(
                    label="Image Number",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=settings["image_number"],
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    show_label=True,
                    placeholder="Type prompt here.",
                    value=settings["negative_prompt"],
                )
                seed_random = gr.Checkbox(label="Random Seed", value=settings["seed_random"])
                image_seed = gr.Number(
                    label="Seed",
                    value=settings["seed"],
                    precision=0,
                    visible=not settings["seed_random"],
                )

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
                    refiner_model = gr.Dropdown(
                        label="SDXL Refiner",
                        choices=["None"] + modules.path.model_filenames,
                        value=settings["refiner_model"],
                        show_label=True,
                    )
                with gr.Accordion(label="LoRAs", open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(
                                label=f"SDXL LoRA {i+1}",
                                choices=["None"] + modules.path.lora_filenames,
                                value=settings[f"lora_{i+1}_model"],
                            )

                            lora_weight = gr.Slider(
                                label="Strength",
                                minimum=-2,
                                maximum=2,
                                step=0.05,
                                value=settings[f"lora_{i+1}_weight"],
                            )
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
                custom_steps = gr.Slider(
                    label="Custom Steps",
                    minimum=10,
                    maximum=200,
                    step=1,
                    value=30,
                    visible=False,
                )
                custom_switch = gr.Slider(
                    label="Custom Switch",
                    minimum=10,
                    maximum=200,
                    step=1,
                    value=20,
                    visible=False,
                )

                cfg = gr.Slider(
                    label="CFG",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    value=8,
                    visible=False,
                )
                base_clip_skip = gr.Slider(
                    label="Base CLIP Skip",
                    minimum=-10,
                    maximum=-1,
                    step=1,
                    value=-2,
                    visible=False,
                )
                refiner_clip_skip = gr.Slider(
                    label="Refiner CLIP Skip",
                    minimum=-10,
                    maximum=-1,
                    step=1,
                    value=-2,
                    visible=False,
                )
                sampler_name = gr.Dropdown(
                    label="Sampler",
                    choices=KSampler.SAMPLERS,
                    value="dpmpp_2m_sde_gpu",
                    visible=False,
                )
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=KSampler.SCHEDULERS,
                    value="karras",
                    visible=False,
                )

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

        ctrls = [
            prompt,
            negative_prompt,
            style_selection,
            performance_selection,
            aspect_ratios_selection,
            image_number,
            image_seed,
            save_metadata,
            cfg,
            base_clip_skip,
            refiner_clip_skip,
            sampler_name,
            scheduler,
            custom_steps,
            custom_switch,
        ]

        ctrls += [base_model, refiner_model] + lora_ctrls
        run_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed).then(
            fn=generate_clicked,
            inputs=ctrls,
            outputs=[run_button, stop_button, progress_html, progress_window, metadata_viewer, gallery],
        )

        def stop_clicked():
            worker.interrupt_ruined_processing = True

        stop_button.click(fn=stop_clicked, queue=False)

args = parse_args()
launch_app(args)
