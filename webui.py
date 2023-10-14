import argparse
import shared
from shared import state, add_ctrl
import time

import gradio as gr

import version
import modules.async_worker as worker
import modules.html
import modules.path
import ui_onebutton
import ui_controlnet

from comfy.samplers import KSampler
from modules.sdxl_styles import style_keys, aspect_ratios, styles
from modules.performance import (
    performance_options,
    load_performance,
    save_performance,
    NEWPERF,
)
from modules.settings import default_settings
from modules.prompt_processing import get_promptlist


from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
    parser.add_argument(
        "--share", action="store_true", help="Set whether to share on Gradio."
    )
    parser.add_argument(
        "--listen",
        type=str,
        default=None,
        metavar="IP",
        nargs="?",
        const="0.0.0.0",
        help="Set the listen interface.",
    )
    parser.add_argument(
        "--nobrowser", action="store_true", help="Do not launch in browser."
    )
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


def update_clicked():
    return {
        run_button: gr.update(interactive=False, visible=False),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True,
            value=modules.html.make_progress_html(0, "Processing text encoding ..."),
        ),
        progress_window: gr.update(visible=True, value="init_image.png"),
    }


def update_preview(product):
    percentage, title, image = product
    return {
        run_button: gr.update(interactive=False, visible=False),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True, value=modules.html.make_progress_html(percentage, title)
        ),
        progress_window: gr.update(visible=True, value=image)
        if image is not None
        else gr.update(),
    }


def update_results(product):
    return {
        run_button: gr.update(interactive=True, visible=True),
        stop_button: gr.update(interactive=False, visible=False),
        progress_html: gr.update(visible=False),
        progress_window: gr.update(value=product[0])
        if len(product) > 0
        else gr.update(),
        gallery: gr.update(allow_preview=True, preview=True, value=product),
    }


def update_done():
    return {
        run_button: gr.update(interactive=True, visible=True),
        stop_button: gr.update(interactive=False, visible=False),
        progress_html: gr.update(visible=False),
        progress_window: gr.update(),
        gallery: gr.update(allow_preview=True, preview=True),
    }


def join_queue(gen_data):
    if not gen_data["session_id"] in worker.buffer:
        worker.buffer[gen_data["session_id"]] = []

    if gen_data["image_number"] > 0:
        yield update_clicked()
        prompts = get_promptlist(gen_data)
        idx = 0

        worker.buffer[gen_data["session_id"]].append(
            {"task_type": "start", "image_total": len(prompts) * gen_data["image_number"]}
        )

        for prompt in prompts:
            gen_data["task_type"] = "process"
            gen_data["prompt"] = prompt
            gen_data["index"] = (idx, len(prompts))
            idx += 1
            worker.buffer[gen_data["session_id"]].append(gen_data.copy())

    if gen_data["image_number"] == 0 and (
    not isinstance(worker.outputs, dict) or
    not gen_data["session_id"] in worker.outputs or
    len(worker.outputs[gen_data["session_id"]]) == 0):
        finished = True
    else:
        finished = False
    while not finished:
        time.sleep(0.1)

        if (not isinstance(worker.outputs, dict) or
        not gen_data["session_id"] in worker.outputs or
        len(worker.outputs[gen_data["session_id"]]) == 0):
            continue

        flag, product = worker.outputs[gen_data["session_id"]].pop(0)

        if flag == "preview":
            yield update_preview(product)

        elif flag == "results":
            yield update_results(product)
            finished = True

    yield update_done()
    worker.buffer[gen_data["session_id"]].append({"task_type": "stop"})


settings = default_settings

if settings["theme"] == "None":
    theme = gr.themes.Default()
else:
    theme = settings["theme"]

shared.gradio_root = gr.Blocks(
    title="RuinedFooocus " + version.version,
    theme=theme,
    css=modules.html.css,
    analytics_enabled=False,
).queue()

with shared.gradio_root as block:
    block.load(_js=modules.html.scripts)
    with gr.Row():
        with gr.Column(scale=5):
            progress_window = gr.Image(
                value="init_image.png",
                height=680,
                type="filepath",
                visible=True,
                show_label=False,
                image_mode="RGBA",
                show_share_button=True,
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
                show_download_button=False,
            )

            @gallery.select(
                inputs=[gallery], outputs=[progress_window], show_progress="hidden"
            )
            def gallery_change(files, sd: gr.SelectData):
                return files[sd.index]["name"]

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
                    run_button = gr.Button(
                        label="Generate", value="Generate", elem_id="generate"
                    )
                    stop_button = gr.Button(
                        label="Stop", value="Stop", interactive=False, visible=False
                    )

                    @progress_window.upload(
                        inputs=[progress_window], outputs=[prompt, gallery]
                    )
                    def load_images_handler(file):
                        image = Image.open(file)
                        info = image.info
                        params = info.get("parameters", "")
                        return params, [file]

            with gr.Row():
                advanced_checkbox = gr.Checkbox(
                    label="Hurt me plenty",
                    value=settings["advanced_mode"],
                    container=False,
                )
        with gr.Column(scale=2, visible=settings["advanced_mode"]) as right_col:
            with gr.Tab(label="Setting"):
                performance_selection = gr.Dropdown(
                    label="Performance",
                    choices=list(performance_options.keys()) + [NEWPERF],
                    value=settings["performance"],
                )
                add_ctrl("performance_selection", performance_selection)
                perf_name = gr.Textbox(
                    show_label=False,
                    placeholder="Name",
                    interactive=True,
                    visible=False,
                )
                perf_save = gr.Button(
                    value="Save",
                    visible=False,
                )
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
                    perf_name,
                    perf_save,
                    cfg,
                    base_clip_skip,
                    refiner_clip_skip,
                    sampler_name,
                    scheduler,
                    custom_steps,
                    custom_switch,
                ]

                @performance_selection.change(
                    inputs=[performance_selection],
                    outputs=[perf_name] + performance_outputs,
                )
                def performance_changed(selection):
                    if selection != NEWPERF:
                        return [gr.update(visible=False)] + [
                            gr.update(visible=False)
                        ] * len(performance_outputs)
                    else:
                        return [gr.update(value="")] + [gr.update(visible=True)] * len(
                            performance_outputs
                        )

                @perf_save.click(
                    inputs=performance_outputs,
                    outputs=[performance_selection],
                )
                def performance_save(
                    perf_name,
                    perf_save,
                    cfg,
                    base_clip_skip,
                    refiner_clip_skip,
                    sampler_name,
                    scheduler,
                    custom_steps,
                    custom_switch,
                ):
                    if perf_name != "":
                        perf_options = load_performance()
                        opts = {
                            "custom_steps": custom_steps,
                            "custom_switch": custom_switch,
                            "cfg": cfg,
                            "base_clip_skip": base_clip_skip,
                            "refiner_clip_skip": refiner_clip_skip,
                            "sampler_name": sampler_name,
                            "scheduler": scheduler,
                        }
                        perf_options[perf_name] = opts
                        save_performance(perf_options)
                        choices = list(perf_options.keys()) + [NEWPERF]
                        return gr.update(choices=choices, value=perf_name)
                    else:
                        return gr.update()

                with gr.Group():
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
                seed_random = gr.Checkbox(
                    label="Random Seed", value=settings["seed_random"]
                )
                image_seed = gr.Number(
                    label="Seed",
                    value=settings["seed"],
                    precision=0,
                    visible=not settings["seed_random"],
                )
                add_ctrl("seed", image_seed)

                @style_button.click(
                    inputs=[prompt, style_selection],
                    outputs=[prompt, negative_prompt, style_selection],
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

                @seed_random.change(inputs=[seed_random], outputs=[image_seed])
                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, s):
                    if r:
                        return -1
                    else:
                        return s

                with gr.Row():
                    session_id = gr.Textbox(
                        label="User/Session",
                        show_label=True,
                        value="none",
                        scale=10,
                    )
                    add_ctrl("session_id", session_id)
                    session_btn = gr.Button(
                        value="🔗",
                        scale=1,
                    )

            with gr.Tab(label="Models"):
                with gr.Row():
                    base_model = gr.Dropdown(
                        label="SDXL Base Model",
                        choices=modules.path.model_filenames,
                        value=settings["base_model"]
                        if settings["base_model"] in modules.path.model_filenames
                        else modules.path.model_filenames[0],
                        show_label=True,
                    )
                    add_ctrl("base_model_name", base_model)
                    refiner_model = gr.Dropdown(
                        label="SDXL Refiner",
                        choices=["None"] + modules.path.model_filenames,
                        value=settings["refiner_model"]
                        if settings["refiner_model"] in modules.path.model_filenames
                        else "None",
                        show_label=True,
                    )
                    add_ctrl("refiner_model_name", refiner_model)
                with gr.Accordion(label="LoRA / Strength", open=True), gr.Group():
                    lora_ctrls = []
                    nones = 0
                    for i in range(5):
                        if settings[f"lora_{i+1}_model"] == "None":
                            nones += 1
                            visible = False if nones > 1 else True
                        else:
                            visible = True
                        with gr.Row():
                            lora_model = gr.Dropdown(
                                label=f"SDXL LoRA {i+1}",
                                show_label=False,
                                choices=["None"] + modules.path.lora_filenames,
                                value=settings[f"lora_{i+1}_model"],
                                visible=visible,
                            )
                            add_ctrl(f"l{i+1}", lora_model)

                            lora_weight = gr.Slider(
                                label="Strength",
                                show_label=False,
                                minimum=-2,
                                maximum=2,
                                step=0.05,
                                value=settings[f"lora_{i+1}_weight"],
                                visible=visible,
                            )
                            add_ctrl(f"w{i+1}", lora_weight)
                            lora_ctrls += [lora_model, lora_weight]

                    def update_loras(*lora_ctrls):
                        hide = False
                        ret = []
                        for m, s in zip(lora_ctrls[::2], lora_ctrls[1::2]):
                            if m == "None":
                                if hide:
                                    ret += [
                                        gr.update(visible=False),
                                        gr.update(visible=False),
                                    ]
                                else:
                                    ret += [
                                        gr.update(visible=True),
                                        gr.update(visible=True),
                                    ]
                                hide = True
                            else:
                                ret += [
                                    gr.update(visible=True),
                                    gr.update(visible=True),
                                ]
                        return ret

                    for m, s in zip(lora_ctrls[::2], lora_ctrls[1::2]):
                        m.change(fn=update_loras, inputs=lora_ctrls, outputs=lora_ctrls)

                with gr.Row():
                    model_refresh = gr.Button(
                        label="Refresh",
                        value="\U0001f504 Refresh All Files",
                        variant="secondary",
                        elem_classes="refresh_button",
                    )

            @model_refresh.click(
                inputs=[], outputs=[base_model, refiner_model] + lora_ctrls
            )
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

            ui_onebutton.ui_onebutton(prompt)

            ui_controlnet.add_controlnet_tab()

        advanced_checkbox.change(
            lambda x: gr.update(visible=x), advanced_checkbox, right_col
        )

        gen_data = gr.State()
        def generate_clicked(*args):
            gen_data = {}
            for key, val in zip(state["ctrls_name"], args):
                gen_data[key] = val
            return gen_data

        run_button.click(
            fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed
        ).then(
            fn=generate_clicked,
            inputs=state["ctrls_obj"],
            outputs=gen_data
        ).then(
            fn=join_queue,
            inputs=gen_data,
            outputs=[run_button, stop_button, progress_html, progress_window, gallery],
        )

        def stop_clicked():
            worker.buffer = []
            worker.interrupt_ruined_processing = True

        stop_button.click(fn=stop_clicked, queue=False)

        def connect_session(sid):
            return [sid, {"image_number": 0, "session_id": sid}]
        session_btn.click(
            fn=connect_session, inputs=[session_id], outputs=[session_id, gen_data]
        ).then(
            fn=join_queue,
            inputs=gen_data,
            outputs=[run_button, stop_button, progress_html, progress_window, gallery],
        )
        session_id.submit(
            fn=connect_session, inputs=[session_id], outputs=[session_id, gen_data]
        ).then(
            fn=join_queue,
            inputs=gen_data,
            outputs=[run_button, stop_button, progress_html, progress_window, gallery],
        )

args = parse_args()
launch_app(args)
