import argparse
import shared
import os
from shared import (
    state,
    add_ctrl,
    performance_settings,
    resolution_settings,
    path_manager,
)
import time

import gradio as gr
import random
import re

import version
import modules.async_worker as worker
import modules.html
import modules.hints
import ui_onebutton
import ui_controlnet
from modules.interrogate import look

from comfy.samplers import KSampler
from modules.sdxl_styles import load_styles, styles, allstyles, apply_style
from modules.settings import default_settings
from modules.prompt_processing import get_promptlist
from modules.util import get_wildcard_files, load_keywords, get_checkpoint_thumbnail
from modules.path import PathManager

from PIL import Image

inpaint_toggle = None
path_manager = PathManager()


def find_unclosed_markers(s):
    markers = re.findall(r"__", s)
    for marker in markers:
        if s.count(marker) % 2 != 0:
            return s.split(marker)[-1]
    return None


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
    parser.add_argument(
        "--share", action="store_true", help="Set whether to share on Gradio."
    )
    parser.add_argument("--auth", type=str, help="Set credentials username/password.")
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
        auth=(
            args.auth.split("/", 1)
            if isinstance(args.auth, str) and "/" in args.auth
            else None
        ),
        favicon_path=favicon_path,
    )


def update_clicked():
    return {
        run_button: gr.update(interactive=False, visible=False),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True,
            value=modules.html.make_progress_html(0, "Please wait ..."),
        ),
        gallery: gr.update(visible=False),
        main_view: gr.update(visible=True, value="init_image.png"),
        inpaint_view: gr.update(visible=False),
        hint_text: gr.update(
            visible=True,
            value=modules.hints.get_hint()
        ),
    }


def update_preview(product):
    percentage, title, image = product
    return {
        run_button: gr.update(interactive=False, visible=False),
        stop_button: gr.update(interactive=True, visible=True),
        progress_html: gr.update(
            visible=True, value=modules.html.make_progress_html(percentage, title)
        ),
        main_view: (
            gr.update(visible=True, value=image) if image is not None else gr.update()
        ),
    }


def update_results(product):
    metadata = {"Data": "Preview Grid"}
    if len(product) > 0:
        with Image.open(product[0]) as im:
            if im.info.get("parameters"):
                metadata = im.info["parameters"]

    return {
        run_button: gr.update(interactive=True, visible=True),
        stop_button: gr.update(interactive=False, visible=False),
        progress_html: gr.update(visible=False),
        main_view: gr.update(value=product[0]) if len(product) > 0 else gr.update(),
        inpaint_toggle: gr.update(value=False),
        gallery: gr.update(
            visible=True, allow_preview=True, preview=True, value=product
        ),
        metadata_json: gr.update(value=metadata),
    }


def append_work(gen_data):
    tmp_data = gen_data.copy()
    if tmp_data["obp_assume_direct_control"]:
        prompts = []
        prompts.append("")
    else:
        prompts = get_promptlist(tmp_data)
    idx = 0

    worker.buffer.append(
        {"task_type": "start", "image_total": len(prompts) * tmp_data["image_number"]}
    )

    for prompt in prompts:
        tmp_data["task_type"] = "process"
        tmp_data["prompt"] = prompt
        tmp_data["index"] = (idx, len(prompts))
        idx += 1
        worker.buffer.append(tmp_data.copy())


def generate_clicked(*args):
    global status

    yield update_clicked()
    gen_data = {}
    for key, val in zip(state["ctrls_name"], args):
        gen_data[key] = val

    if int(gen_data["image_number"]) == 0:
        generate_forever = True
    else:
        generate_forever = False
    gen_data["generate_forever"] = generate_forever

    append_work(gen_data)

    shared.state["interrupted"] = False
    finished = False
    while not finished:
        time.sleep(0.1)

        if not worker.outputs:
            continue

        flag, product = worker.outputs.pop(0)

        if flag == "preview":
            yield update_preview(product)

        elif flag == "results":
            worker.buffer.append({"task_type": "stop"})

            if generate_forever and shared.state["interrupted"] == False:
                append_work(gen_data)
            else:
                yield update_results(product)
                finished = True

    shared.state["interrupted"] = False


def calculateTokenCounter(text):
    if len(text) < 1:
        return 0
    return len(shared.tokenizer.tokenize(text))


settings = default_settings

if settings["theme"] == "None":
    theme = gr.themes.Default()
else:
    theme = settings["theme"]

metadata_json = gr.Json()

shared.wildcards = get_wildcard_files()

shared.gradio_root = gr.Blocks(
    title="RuinedFooocus " + version.version,
    theme=theme,
    css=modules.html.css,
    analytics_enabled=True,
    concurrency_count=4,
).queue()

with shared.gradio_root as block:
    block.load(_js=modules.html.scripts)
    run_event = gr.Number(visible=False, value=0)
    with gr.Row():
        with gr.Column(scale=5):
            main_view = gr.Image(
                value="init_image.png",
                height=680,
                type="filepath",
                visible=True,
                show_label=False,
                image_mode="RGBA",
            )
            add_ctrl("main_view", main_view)
            inpaint_view = gr.Image(
                height=680,
                type="numpy",
                elem_id="inpaint_sketch",
                tool="sketch",
                visible=False,
                image_mode="RGBA",
            )
            add_ctrl("inpaint_view", inpaint_view)

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
                inputs=[gallery],
                outputs=[main_view, metadata_json],
                show_progress="hidden",
            )
            def gallery_change(files, sd: gr.SelectData):
                names = files[sd.index]["name"]
                with Image.open(files[sd.index]["name"]) as im:
                    if im.info.get("parameters"):
                        metadata = im.info["parameters"]
                    else:
                        metadata = {"Data": "Preview Grid"}
                return [names] + [gr.update(value=metadata)]

            with gr.Row(elem_classes="type_row"):
                with gr.Column(scale=5):
                    with gr.Group():
                        with gr.Group(), gr.Row():
                            prompt = gr.Textbox(
                                show_label=False,
                                placeholder="Type prompt here.",
                                container=False,
                                autofocus=True,
                                elem_classes="type_row",
                                lines=1024,
                                value=settings["prompt"],
                                scale=4,
                            )
                            add_ctrl("prompt", prompt)

                            spellcheck = gr.Dropdown(
                                label="Wildcards",
                                visible=False,
                                choices=[],
                                value="",
                                scale=1,
                            )

                        prompt_token_counter = gr.HTML(
                            visible=settings["advanced_mode"],
                            value=str(
                                calculateTokenCounter(settings["prompt"])
                            ),  # start with token count for default prompt
                            elem_classes=["tokenCounter"],
                        )

                    @prompt.change(inputs=prompt, outputs=prompt_token_counter)
                    def updatePromptTokenCount(text):
                        return calculateTokenCounter(text)

                    @prompt.input(inputs=prompt, outputs=spellcheck)
                    def checkforwildcards(text):
                        test = find_unclosed_markers(text)
                        tokencount = len(shared.tokenizer.tokenize(text))
                        if test is not None:
                            filtered = [s for s in shared.wildcards if test in s]
                            filtered.append(" ")
                            return {
                                spellcheck: gr.update(
                                    interactive=True,
                                    visible=True,
                                    choices=filtered,
                                    value=" ",
                                )
                            }
                        else:
                            return {
                                spellcheck: gr.update(interactive=False, visible=False)
                            }

                    @spellcheck.select(inputs=[prompt, spellcheck], outputs=prompt)
                    def select_spellcheck(text, selection):
                        last_idx = text.rindex("__")
                        newtext = f"{text[:last_idx]}__{selection}__"
                        return {prompt: gr.update(value=newtext)}

                with gr.Column(scale=1, min_width=0):
                    run_button = gr.Button(value="Generate", elem_id="generate")
                    stop_button = gr.Button(
                        value="Stop", interactive=False, visible=False
                    )

                    @main_view.upload(
                        inputs=[main_view, prompt], outputs=[prompt, gallery]
                    )
                    def load_images_handler(file, prompt):
                        image = Image.open(file)
                        params = look(image, prompt, gr)
                        return params, [file]

            with gr.Row():
                advanced_checkbox = gr.Checkbox(
                    label="Hurt me plenty",
                    value=settings["advanced_mode"],
                    container=False,
                    elem_id="hurtme",
                )
        with gr.Column(scale=2, visible=settings["advanced_mode"]) as right_col:
            with gr.Tab(label="Setting"):
                performance_selection = gr.Dropdown(
                    label="Performance",
                    choices=list(performance_settings.performance_options.keys())
                    + [performance_settings.CUSTOM_PERFORMANCE],
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
                custom_default_values = performance_settings.get_perf_options(
                    settings["performance"]
                )
                custom_steps = gr.Slider(
                    label="Custom Steps",
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=custom_default_values["custom_steps"],
                    visible=False,
                )
                add_ctrl("custom_steps", custom_steps)

                cfg = gr.Slider(
                    label="CFG",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=custom_default_values["cfg"],
                    visible=False,
                )
                add_ctrl("cfg", cfg)
                sampler_name = gr.Dropdown(
                    label="Sampler",
                    choices=KSampler.SAMPLERS,
                    value=custom_default_values["sampler_name"],
                    visible=False,
                )
                add_ctrl("sampler_name", sampler_name)
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=KSampler.SCHEDULERS,
                    value=custom_default_values["scheduler"],
                    visible=False,
                )
                add_ctrl("scheduler", scheduler)

                clip_skip = gr.Slider(
                    label="Clip Skip",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                    visible=False,
                )

                add_ctrl("clip_skip", clip_skip)

                performance_outputs = [
                    perf_name,
                    perf_save,
                    cfg,
                    sampler_name,
                    scheduler,
                    clip_skip,
                    custom_steps,
                ]

                @perf_save.click(
                    inputs=performance_outputs,
                    outputs=[performance_selection],
                )
                def performance_save(
                    perf_name,
                    perf_save,
                    cfg,
                    sampler_name,
                    scheduler,
                    clip_skip,
                    custom_steps,
                ):
                    if perf_name != "":
                        perf_options = performance_settings.load_performance()
                        opts = {
                            "custom_steps": custom_steps,
                            "cfg": cfg,
                            "sampler_name": sampler_name,
                            "scheduler": scheduler,
                            "clip_skip": clip_skip,
                        }
                        perf_options[perf_name] = opts
                        performance_settings.save_performance(perf_options)
                        choices = list(perf_options.keys()) + [
                            performance_settings.CUSTOM_PERFORMANCE
                        ]
                        return gr.update(choices=choices, value=perf_name)
                    else:
                        return gr.update()

                with gr.Group():
                    aspect_ratios_selection = gr.Dropdown(
                        label="Aspect Ratios (width x height)",
                        choices=list(resolution_settings.aspect_ratios.keys())
                        + [resolution_settings.CUSTOM_RESOLUTION],
                        value=settings["resolution"],
                    )
                    add_ctrl("aspect_ratios_selection", aspect_ratios_selection)
                    ratio_name = gr.Textbox(
                        show_label=False,
                        placeholder="Name",
                        interactive=True,
                        visible=False,
                    )
                    default_resolution = resolution_settings.get_aspect_ratios(
                        settings["resolution"]
                    )
                    custom_width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=4096,
                        step=2,
                        visible=False,
                        value=default_resolution[0],
                    )
                    add_ctrl("custom_width", custom_width)
                    custom_height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=4096,
                        step=2,
                        visible=False,
                        value=default_resolution[1],
                    )
                    add_ctrl("custom_height", custom_height)
                    ratio_save = gr.Button(
                        value="Save",
                        visible=False,
                    )

                    @ratio_save.click(
                        inputs=[ratio_name, custom_width, custom_height],
                        outputs=[aspect_ratios_selection],
                    )
                    def ratio_save_click(ratio_name, custom_width, custom_height):
                        if ratio_name != "":
                            ratio_options = resolution_settings.load_resolutions()
                            ratio_options[ratio_name] = (custom_width, custom_height)
                            resolution_settings.save_resolutions(ratio_options)
                            choices = list(resolution_settings.aspect_ratios.keys()) + [
                                resolution_settings.CUSTOM_RESOLUTION
                            ]
                            new_ratio_name = (
                                f"{custom_width}x{custom_height} ({ratio_name})"
                            )
                            return gr.update(choices=choices, value=new_ratio_name)
                        else:
                            return gr.update()

                    style_selection = gr.Dropdown(
                        label="Style Selection",
                        multiselect=True,
                        container=True,
                        choices=list(load_styles().keys()),
                        value=settings["style"],
                    )
                    add_ctrl("style_selection", style_selection)
                style_button = gr.Button(value="⬅️ Send Style to prompt", size="sm")
                image_number = gr.Slider(
                    label="Image Number",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=settings["image_number"],
                )
                add_ctrl("image_number", image_number)
                auto_negative_prompt = gr.Checkbox(
                    label="Auto Negative Prompt",
                    show_label=True,
                    value=settings["auto_negative_prompt"],
                )
                add_ctrl("auto_negative", auto_negative_prompt)
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
                    inputs=[prompt, negative_prompt, style_selection],
                    outputs=[prompt, negative_prompt, style_selection],
                )
                def send_to_prompt(prompt_text, negative_text, style_inputs):
                    prompt_style, negative_style = apply_style(
                        style_inputs, prompt_text, negative_text, ""
                    )
                    return prompt_style, negative_style, []

                @seed_random.change(inputs=[seed_random], outputs=[image_seed])
                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, s):
                    if r:
                        return -1
                    else:
                        return s

            with gr.Tab(label="Models"):
                if settings["base_model"] not in path_manager.model_filenames:
                    settings["base_model"] = path_manager.model_filenames[0]

                with gr.Row():
                    model_gallery = gr.Gallery(
                        label="SDXL model: sdxl-something.safetensors",
                        show_label=True,
                        object_fit="scale-down",
                        height=300,
                        allow_preview=False,
                        preview=False,
                        visible=True,
                        show_download_button=False,
                        min_width=60,
                        coulmns=3,
                        value=list(map(lambda x: (get_checkpoint_thumbnail(x), x), path_manager.model_filenames)),
                    )

                    base_model = gr.Text(
                        visible=False,
                        value=settings["base_model"],
                    )
                    add_ctrl("base_model_name", base_model)

                    def update_model_select(evt: gr.SelectData):
                        return {
                            model_gallery: gr.update(label=f"SDXL model: {evt.value[1]}"),
                            base_model: gr.update(value=evt.value[1]),
                        }

                    model_gallery.select(
                        update_model_select,
                        None,
                        outputs=[model_gallery, base_model]
                    )

#                with gr.Row():
#                    base_model = gr.Dropdown(
#                        visible=False,
#                        label="SDXL Base Model",
#                        choices=path_manager.model_filenames,
#                        value=(
#                            settings["base_model"]
#                            if settings["base_model"] in path_manager.model_filenames
#                            else [path_manager.model_filenames[0]]
#                        ),
#                        show_label=True,
#                    )
#                    add_ctrl("base_model_name", base_model)
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
                                choices=["None"] + path_manager.lora_filenames,
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

                    lora_keywords = gr.Textbox(
                        label="LoRA Trigger Words", interactive=False
                    )
                    add_ctrl("lora_keywords", lora_keywords)

                    def update_loras_visibility(*lora_controls):
                        """Update the visibility of LoRa controls based on their values."""
                        hide = False
                        updates = []
                        lora_prompt_addition = ""
                        for model, strength in zip(
                            lora_controls[::2], lora_controls[1::2]
                        ):
                            visibility = False if model == "None" and hide else True
                            updates += [
                                gr.update(visible=visibility),
                                gr.update(visible=visibility),
                            ]
                            if model == "None":
                                hide = True
                            if model == "None" or strength == 0:
                                continue
                            filename = os.path.join(
                                path_manager.model_paths["lorafile_path"], model
                            )
                            lora_prompt_addition = (
                                f"{lora_prompt_addition} {load_keywords(filename)} "
                            )

                        updates += [lora_keywords.update(value=lora_prompt_addition)]
                        return updates

                    for model, strength in zip(lora_ctrls[::2], lora_ctrls[1::2]):
                        model.change(
                            fn=update_loras_visibility,
                            inputs=lora_ctrls,
                            outputs=lora_ctrls + [lora_keywords],
                        )

                with gr.Row():
                    model_refresh = gr.Button(
                        value="\U0001f504 Refresh All Files",
                        variant="secondary",
                        elem_classes="refresh_button",
                    )

            @model_refresh.click(
                inputs=[], outputs=[base_model] + lora_ctrls + [style_selection]
            )
            def model_refresh_clicked():
                path_manager.update_all_model_names()
                results = []
                results += [
                    gr.update(choices=path_manager.model_filenames),
                ]
                for i in range(5):
                    results += [
                        gr.update(choices=["None"] + path_manager.lora_filenames),
                        gr.update(),
                    ]
                results += [gr.update(choices=list(load_styles().keys()))]
                return results

            ui_onebutton.ui_onebutton(prompt, run_event)

            inpaint_toggle = ui_controlnet.add_controlnet_tab(
                main_view, inpaint_view, prompt, image_number, run_event
            )

            with gr.Tab(label="Info"):
                with gr.Row():
                    metadata_json.render()
                with gr.Row():
                    gr.HTML(
                        value="""
                        <a href="https://discord.gg/CvpAFya9Rr"><img src="file=html/icon_clyde_white_RGB.svg" height="16" width="16" style="display:inline-block;">&nbsp;Discord</a><br>
                        <a href="https://github.com/runew0lf/RuinedFooocus"><img src="file=html/github-mark-white.svg" height="16" width="16" style="display:inline-block;">&nbsp;Github</a><br>
                        <a href="file=html/slideshow.html" style="color: gray; text-decoration: none" target="_blank">&pi;</a>
                        """,
                    )

            hint_text = gr.Markdown(
                value="",
                elem_id="hint-container",
                elem_classes="hint-container",
            )

            @sampler_name.change(
                inputs=[sampler_name], outputs=[lora_ctrls[0]] + [lora_ctrls[1]]
            )
            def sampler_changed(sampler_name):
                if sampler_name == "lcm":
                    return [gr.update(value=path_manager.find_lcm_lora())] + [
                        gr.update(value=1.0)
                    ]
                else:
                    return [gr.update()] + [gr.update()]

            @performance_selection.change(
                inputs=[performance_selection],
                outputs=[perf_name]
                + performance_outputs
                + [lora_ctrls[0]]
                + [lora_ctrls[1]],
            )
            def performance_changed(selection):
                if selection == performance_settings.CUSTOM_PERFORMANCE:
                    return (
                        [perf_name.update(value="")]
                        + [gr.update(visible=True)] * len(performance_outputs)
                        + [lora_ctrls[0].update()]
                        + [lora_ctrls[1].update()]
                    )
                elif selection == "Lcm":
                    return (
                        [perf_name.update(visible=False)]
                        + [gr.update(visible=False)] * len(performance_outputs)
                        + [lora_ctrls[0].update(value=path_manager.find_lcm_lora())]
                        + [lora_ctrls[1].update(value=1.0)]
                    )

                else:
                    return (
                        [perf_name.update(visible=False)]
                        + [gr.update(visible=False)] * len(performance_outputs)
                        + [lora_ctrls[0].update()]
                        + [lora_ctrls[1].update()]
                    )

            @performance_selection.change(
                inputs=[performance_selection],
                outputs=[custom_steps]
                + [cfg]
                + [sampler_name]
                + [scheduler]
                + [clip_skip],
            )
            def performance_changed_update_custom(selection):
                # Skip if Custom was selected
                if selection == performance_settings.CUSTOM_PERFORMANCE:
                    return [gr.update()] * 5

                # Update Custom values based on selected Performance mode
                selected_perf_options = performance_settings.get_perf_options(selection)
                return [
                    custom_steps.update(value=selected_perf_options["custom_steps"]),
                    cfg.update(value=selected_perf_options["cfg"]),
                    sampler_name.update(value=selected_perf_options["sampler_name"]),
                    scheduler.update(value=selected_perf_options["scheduler"]),
                    clip_skip.update(value=selected_perf_options["clip_skip"]),
                ]

            @aspect_ratios_selection.change(
                inputs=[aspect_ratios_selection],
                outputs=[ratio_name, custom_width, custom_height, ratio_save],
            )
            def aspect_ratios_changed(selection):
                # Show resolution controls when selecting Custom
                if selection == resolution_settings.CUSTOM_RESOLUTION:
                    return [gr.update(visible=True)] * 4

                # Hide resolution controls and update with selected resolution
                selected_width, selected_height = resolution_settings.get_aspect_ratios(
                    selection
                )
                return [
                    ratio_name.update(visible=False),
                    custom_width.update(visible=False, value=selected_width),
                    custom_height.update(visible=False, value=selected_height),
                    ratio_save.update(visible=False),
                ]

        def update_token_visibility(x):
            return [gr.update(visible=x), gr.update(visible=x)]

        advanced_checkbox.change(
            update_token_visibility,
            inputs=advanced_checkbox,
            outputs=[right_col, prompt_token_counter],
        )

        run_event.change(
            fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed
        ).then(
            fn=generate_clicked,
            inputs=state["ctrls_obj"],
            outputs=[
                run_button,
                stop_button,
                progress_html,
                main_view,
                inpaint_view,
                inpaint_toggle,
                gallery,
                metadata_json,
                hint_text,
            ],
        )

        def poke(number):
            return number + 1

        run_button.click(fn=poke, inputs=run_event, outputs=run_event)

        def stop_clicked():
            worker.buffer = []
            worker.interrupt_ruined_processing = True
            shared.state["interrupted"] = False

        stop_button.click(fn=stop_clicked, queue=False)

    def get_last_image():
        global state
        if "last_image" in state:
            return state["last_image"]
        else:
            return "logo.png"

    last_image = gr.Button(visible=False)
    last_image.click(get_last_image, outputs=[last_image], api_name="last_image")

args = parse_args()
if isinstance(args.auth, str) and not "/" in args.auth:
    if len(args.auth):
        print(
            f'\nERROR! --auth need be in the form of "username/password" not "{args.auth}"\n'
        )
    if args.share:
        print(
            f"\nWARNING! Will not enable --share without proper --auth=username/password\n"
        )
        args.share = False
launch_app(args)
