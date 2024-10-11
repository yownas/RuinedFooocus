import argparse
import shared
from pathlib import Path
from shared import (
    state,
    add_ctrl,
    performance_settings,
    resolution_settings,
    path_manager,
)
import time
import json
import gradio as gr
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
from modules.util import (
    get_wildcard_files,
    load_keywords,
    get_checkpoint_thumbnail,
    get_lora_thumbnail,
    get_model_thumbnail,
)
from modules.path import PathManager
from modules.civit import Civit

from PIL import Image

inpaint_toggle = None
path_manager = PathManager()
civit_checkpoints = Civit(
    model_dir=Path(path_manager.model_paths["modelfile_path"]),
    cache_path=Path(path_manager.model_paths["cache_path"] / "checkpoints")
)
civit_loras = Civit(
    model_dir=Path(path_manager.model_paths["lorafile_path"]),
    cache_path=Path(path_manager.model_paths["cache_path"] / "loras")
)

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
    shared.gradio_root.queue()
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
        hint_text: gr.update(visible=True, value=modules.hints.get_hint()),
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

    # FIXME this is _ugly_ run_event gets triggerd once at page load
    #   not really gradios fault, we are doing silly things there. :)
    print(f"DEBUG: it got clicked {gen_data['run_event']}")
    print(gen_data["run_event"])
    if gen_data["run_event"] < 1:
        yield update_results(["logo.png"])
        return

    if int(gen_data["image_number"]) == -1:
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

            if generate_forever and shared.state["interrupted"] == False:
                worker.buffer.append({"task_type": "stop"})
                append_work(gen_data)
            else:
                yield update_results(product)
                finished = True

    worker.buffer.append({"task_type": "stop"})
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
    analytics_enabled=False,
).queue()

with shared.gradio_root as block:
    block.load(js=modules.html.scripts)
    run_event = gr.Number(visible=False, value=0)
    add_ctrl("run_event", run_event)
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
                visible=False,
                image_mode="RGBA",
            )
            # FIxME    tool="sketch",
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
                name = sd.value["image"]["path"]
                with Image.open(name) as im:
                    if im.info.get("parameters"):
                        metadata = im.info["parameters"]
                    else:
                        metadata = {"Data": "Preview Grid"}
                return [name] + [gr.update(value=metadata)]

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
                    # FIXME run_button = gr.Button(value="Generate", elem_id="generate", api_name="generate")
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

                with gr.Tab(label="Model"):
                    model_current = gr.HTML(
                        value=f"{settings['base_model']}",
                    )
                    # FIXME    container=False,
                    # FIXME    interactive=False,
                    with gr.Group():
                        modelfilter = gr.Textbox(
                            placeholder="Model name",
                            value="",
                            show_label=False,
                            container=False,
                        )
                        model_gallery = gr.Gallery(
                            label=f"SDXL model: {settings['base_model']}",
                            show_label=False,
                            object_fit="scale-down",
                            height=550,
                            allow_preview=False,
                            preview=False,
                            visible=True,
                            show_download_button=False,
                            min_width=60,
                            value=list(
                                map(
                                    lambda x: (get_checkpoint_thumbnail(x), x),
                                    path_manager.model_filenames,
                                )
                            ),
                        )

                    base_model = gr.Text(
                        visible=False,
                        value=settings["base_model"],
                    )
                    add_ctrl("base_model_name", base_model)

                    @modelfilter.input(inputs=modelfilter, outputs=[model_gallery])
                    def update_model_filter(filtered):
                        filtered_filenames = filter(
                            lambda filename: filtered.lower() in filename.lower(),
                            path_manager.model_filenames,
                        )
                        newlist = list(
                            map(
                                lambda x: (get_checkpoint_thumbnail(x), x),
                                filtered_filenames,
                            )
                        )
                        return gr.update(value=newlist)

                    def update_model_select(evt: gr.SelectData):
                        model_name = f"{evt.value['caption']}"
                        models = civit_checkpoints.get_models_by_path(path_manager.model_paths["modelfile_path"] / Path(model_name))
                        model_base = civit_checkpoints.get_model_base(models)

                        txt = f"{evt.value['caption']}<br>Model type: {model_base}"

                        return {
                            model_current: gr.update(value=txt),
                            base_model: gr.update(value=model_name),
                        }

                    model_gallery.select(
                        update_model_select, None, outputs=[model_current, base_model]
                    )

                with gr.Tab(label="LoRAs"):
                    with gr.Group(visible=False) as lora_add:
                        lorafilter = gr.Textbox(
                            placeholder="Search LoRA",
                            value="",
                            show_label=False,
                            container=False,
                        )
                        lora_gallery = gr.Gallery(
                            label=f"LoRA model",
                            show_label=False,
                            object_fit="scale-down",
                            height=510,
                            allow_preview=False,
                            preview=False,
                            show_download_button=False,
                            min_width=60,
                            value=list(
                                map(
                                    lambda x: (get_lora_thumbnail(x), x),
                                    path_manager.lora_filenames,
                                )
                            ),
                        )
                        lora_cancel_btn = gr.Button(
                            value="cancel",
                            scale=1,
                        )

                    default_active = []
                    for i in range(1,6):
                        m = default_settings.get(f"lora_{i}_model", "None")
                        w = default_settings.get(f"lora_{i}_weight", 0.0)
                        if m != "" and m != "None":
                            default_active.append((get_lora_thumbnail(m), f"{w} - {m}"))

                    with gr.Group(visible=True) as lora_active:
                        with gr.Row():
                            lora_weight_slider = gr.Slider(
                                label="Weight",
                                show_label=True,
                                minimum=0,
                                maximum=2,
                                step=0.05,
                                value=1.0,
                                interactive=True,
                            )
                        lora_active_gallery = gr.Gallery(
                            label=f"LoRA model",
                            show_label=False,
                            object_fit="scale-down",
                            height=510,
                            allow_preview=False,
                            preview=False,
                            visible=True,
                            show_download_button=False,
                            min_width=60,
                            value=default_active,
                        )
                        add_ctrl("loras", lora_active_gallery)

                        with gr.Group(), gr.Row():
                            lora_add_btn = gr.Button(
                                value="+",
                                scale=1,
                            )
                            
                            lora_del_btn = gr.Button(
                                value="-",
                                scale=1,
                            )

                        lora_keywords = gr.Textbox(
                            label="LoRA Trigger Words", interactive=False
                        )
                        add_ctrl("lora_keywords", lora_keywords)

                with gr.Tab(label="MergeMaker"):
                    with gr.Group():
                        mm_name = gr.Textbox(
                            show_label=False,
                            placeholder="Name(.merge)",
                            container=False,
                        )
                        mm_comment = gr.Textbox(
                            show_label=False,
                            placeholder="Comment",
                            container=False,
                        )
                        mm_cache = gr.Checkbox(
                            label="Save cached safetensor of merged model",
                            value=False,
                            container=False,
                        )

                    with gr.Group(visible=False) as mm_add:
                        mm_filter = gr.Textbox(
                            placeholder="Search model/LoRA",
                            value="",
                            show_label=False,
                            container=False,
                        )
                        mm_gallery = gr.Gallery(
                            label=f"Models",
                            show_label=False,
                            object_fit="scale-down",
                            height=410,
                            allow_preview=False,
                            preview=False,
                            show_download_button=False,
                            min_width=60,
                            value=list(
                                map(
                                    lambda x: (get_checkpoint_thumbnail(x), f"C:{x}"),
                                    path_manager.model_filenames,
                                )
                            ) + list (
                                map(
                                    lambda x: (get_lora_thumbnail(x), f"L:{x}"),
                                    path_manager.lora_filenames,
                                )
                            ),
                        )
                        mm_cancel_btn = gr.Button(
                            value="cancel",
                            scale=1,
                        )

                    with gr.Group(visible=True) as mm_active:
                        with gr.Row():
                            mm_weight_slider = gr.Slider(
                                label="Weight",
                                show_label=True,
                                minimum=0,
                                maximum=2,
                                step=0.05,
                                value=1.0,
                                interactive=True,
                            )
                        mm_active_gallery = gr.Gallery(
                            label=f"Models",
                            show_label=False,
                            object_fit="scale-down",
                            height=410,
                            allow_preview=False,
                            preview=False,
                            visible=True,
                            show_download_button=False,
                            min_width=60,
                            value=[],
                        )
                        add_ctrl("mm_models", mm_active_gallery)

                        with gr.Group(), gr.Row():
                            mm_add_btn = gr.Button(
                                value="+",
                                scale=1,
                            )
                            mm_del_btn = gr.Button(
                                value="-",
                                scale=1,
                            )

                        mm_save_btn = gr.Button(
                            value="Save",
                        )

                def gallery_toggle():
                    result = [
                        gr.update(visible=True),
                        gr.update(visible=False),
                    ]
                    return result

                # LoRA
                @lorafilter.input(inputs=lorafilter, outputs=[lora_gallery])
                def update_lora_filter(filtered):
                    filtered_filenames = filter(
                        lambda filename: filtered.lower() in filename.lower(),
                        path_manager.lora_filenames,
                    )
                    newlist = list(
                        map(
                            lambda x: (get_lora_thumbnail(x), x),
                            filtered_filenames,
                        )
                    )
                    return gr.update(value=newlist)

                def lora_select(gallery, evt: gr.SelectData):
                    w = 1.0

                    keywords = ""
                    loras = []
                    if gallery is not None:
                        for lora_data in gallery:
                            w, l = lora_data[1].split(" - ", 1)
                            keywords = f"{keywords}, {load_keywords(l)} "

                            loras.append((lora_data[0], lora_data[1]))
                    keywords = f"{keywords}, {load_keywords(evt.value['caption'])} "

                    loras.append(
                        (get_lora_thumbnail(evt.value['caption']), f"{w} - {evt.value['caption']}")
                    )
                    return {
                        lora_add: gr.update(visible=False),
                        lora_active: gr.update(visible=True),
                        lora_active_gallery: gr.update(value=loras),
                        lora_keywords: gr.update(value=keywords),
                    }

                lora_active_selected = None

                def lora_active_select(gallery, evt: gr.SelectData):
                    global lora_active_selected
                    lora_active_selected = evt.index
                    return {
                        lora_active: gr.update(),
                        lora_active_gallery: gr.update(),
                        lora_weight_slider: gr.update(
                            value=float(evt.value['caption'].split(" - ", 1)[0])
                        ),
                    }

                def lora_delete(gallery):
                    global lora_active_selected
                    if lora_active_selected is not None:
                        del gallery[lora_active_selected]
                        if lora_active_selected >= len(gallery):
                            lora_active_selected = None
                    keywords = ""
                    loras = []
                    for lora_data in gallery:
                        w, l = lora_data[1].split(" - ", 1)
                        loras.append((lora_data[0], lora_data[1]))
                        keywords = f"{keywords}, {load_keywords(l)} "
                    return {
                        lora_active_gallery: gr.update(value=loras),
                        lora_keywords: gr.update(value=keywords),
                    }

                def lora_weight_slider_update(gallery, w):
                    global lora_active_selected
                    if lora_active_selected is None:
                        return {lora_active_gallery: gr.update()}

                    loras = []
                    for lora_data in gallery:
                        loras.append((lora_data[0], lora_data[1]))
                    l = gallery[lora_active_selected][1].split(" - ")[1]
                    loras[lora_active_selected] = (get_lora_thumbnail(l), f"{w} - {l}")

                    return {
                        lora_active_gallery: gr.update(value=loras),
                    }

                lora_weight_slider.release(
                    fn=lora_weight_slider_update,
                    inputs=[lora_active_gallery, lora_weight_slider],
                    outputs=[lora_active_gallery],
                )
                lora_add_btn.click(
                    fn=gallery_toggle,
                    outputs=[lora_add, lora_active],
                )
                lora_cancel_btn.click(
                    fn=gallery_toggle,
                    outputs=[lora_active, lora_add],
                )
                lora_del_btn.click(
                    fn=lora_delete,
                    inputs=lora_active_gallery,
                    outputs=[lora_active_gallery, lora_keywords],
                )
                lora_gallery.select(
                    fn=lora_select,
                    inputs=[lora_active_gallery],
                    outputs=[lora_add, lora_active, lora_active_gallery, lora_keywords],
                )
                lora_active_gallery.select(
                    fn=lora_active_select,
                    inputs=[lora_active_gallery],
                    outputs=[lora_active, lora_active_gallery, lora_weight_slider],
                )

                # MM
                @mm_filter.input(inputs=mm_filter, outputs=[mm_gallery])
                def update_mm_filter(filtered):
                    filtered_models = filter(
                        lambda filename: ".merge" not in filename.lower(),
                        path_manager.model_filenames,
                    )
                    filtered_models = filter(
                        lambda filename: filtered.lower() in filename.lower(),
                        filtered_models,
                    )
                    filtered_loras = filter(
                        lambda filename: filtered.lower() in filename.lower(),
                        path_manager.lora_filenames,
                    )
                    newlist = list(
                        map(
                            lambda x: (get_checkpoint_thumbnail(x), f"C:{x}"),
                            filtered_models,
                        )
                    ) + list (
                        map(
                            lambda x: (get_lora_thumbnail(x), f"L:{x}"),
                            filtered_loras,
                        )
                    )
                    return gr.update(value=newlist)


                def mm_select(gallery, evt: gr.SelectData):
                    w = 1.0

                    mm = []
                    for mm_data in gallery:
                        mm.append((mm_data[0]["name"], mm_data[1]))

                    m = evt.value[1]
                    n = re.sub("[CL]:", "", m)
                    mm.append(
                        (get_model_thumbnail(n), f"{w} - {m}")
                    )
                    return {
                        mm_add: gr.update(visible=False),
                        mm_active: gr.update(visible=True),
                        mm_active_gallery: gr.update(value=mm),
                    }

                mm_active_selected = None

                def mm_active_select(gallery, evt: gr.SelectData):
                    global mm_active_selected
                    mm_active_selected = evt.index
                    mm = []
                    for mm_data in gallery:
                        mm.append((mm_data[0]["name"], mm_data[1]))
                    return {
                        mm_active: gr.update(),
                        mm_active_gallery: gr.update(),
                        mm_weight_slider: gr.update(
                            value=float(mm[evt.index][1].split(" - ", 1)[0])
                        ),
                    }


                def mm_delete(gallery):
                    global mm_active_selected
                    if mm_active_selected is not None:
                        del gallery[mm_active_selected]
                        if mm_active_selected >= len(gallery):
                            mm_active_selected = None
                    mm = []
                    for mm_data in gallery:
                        w, l = mm_data[1].split(" - ", 1)
                        mm.append((mm_data[0]["name"], mm_data[1]))
                    return {
                        mm_active_gallery: gr.update(value=mm),
                    }

                def mm_weight_slider_update(gallery, w):
                    global mm_active_selected
                    if mm_active_selected is None:
                        return {mm_active_gallery: gr.update()}

                    mm = []
                    for mm_data in gallery:
                        mm.append((mm_data[0]["name"], mm_data[1]))
                    l = gallery[mm_active_selected][1].split(" - ")[1]
                    n = re.sub("[CL]:", "", l)
                    mm[mm_active_selected] = (get_model_thumbnail(n), f"{w} - {l}")

                    return {
                        mm_active_gallery: gr.update(value=mm),
                    }

                def mm_save(name, comment, gallery, cache):
                    if name == "":
                        gr.Info("Merge needs a name.")
                        return
                    if comment == "":
                        gr.Info("You probably want a comment.")
                        return

                    dict = {}
                    models = []
                    loras = []

                    for model_data in gallery:
                        w, m = model_data[1].split(" - ", 1)
                        n = re.sub("[CL]:", "", m)
                        if m.startswith("C:"):
                            models.append((n, w))
                        if m.startswith("L:"):
                            loras.append((n, w))

                    dict["comment"] = comment
                    base = models.pop(0)
                    dict["base"] = {"name": base[0], "weight": float(base[1])}
                    dict["models"] = []
                    for model in models:
                        dict["models"].append({"name": model[0], "weight": float(model[1])})
                    dict["loras"] = []
                    for lora in loras:
                        dict["loras"].append({"name": lora[0], "weight": float(lora[1])})
                    dict["normalize"] = 1.0
                    dict["cache"] = cache

                    filename = Path(path_manager.model_paths["modelfile_path"] / name).with_suffix(".merge")
                    if filename.exists():
                        gr.Info("Not saving, file already exists.")
                    else:
                        with open(filename, "w") as outfile: 
                            json.dump(dict, outfile, indent=2)
                        gr.Info(f"Saved {Path(name).with_suffix('.merge')}")


                mm_weight_slider.release(
                    fn=mm_weight_slider_update,
                    inputs=[mm_active_gallery, mm_weight_slider],
                    outputs=[mm_active_gallery],
                )
                mm_add_btn.click(
                    fn=gallery_toggle,
                    outputs=[mm_add, mm_active],
                )
                mm_cancel_btn.click(
                    fn=gallery_toggle,
                    outputs=[mm_active, mm_add],
                )
                mm_del_btn.click(
                    fn=mm_delete,
                    inputs=mm_active_gallery,
                    outputs=[mm_active_gallery],
                )
                mm_gallery.select(
                    fn=mm_select,
                    inputs=[mm_active_gallery],
                    outputs=[mm_add, mm_active, mm_active_gallery],
                )
                mm_active_gallery.select(
                    fn=mm_active_select,
                    inputs=[mm_active_gallery],
                    outputs=[mm_active, mm_active_gallery, mm_weight_slider],
                )
                mm_save_btn.click(
                    fn=mm_save,
                    inputs=[mm_name, mm_comment, mm_active_gallery, mm_cache],
                    outputs=[],
                )

                with gr.Row():
                    model_refresh = gr.Button(
                        value="\U0001f504 Refresh All Files",
                        variant="secondary",
                        elem_classes="refresh_button",
                    )

            @model_refresh.click(
                inputs=[],
                outputs=[modelfilter, model_gallery, lorafilter, lora_gallery, style_selection, mm_filter, mm_gallery]
            )
            def model_refresh_clicked():
                path_manager.update_all_model_names()

                # model_filter
                results = [gr.update(value="")]
                # model_gallery
                results += [update_model_filter("")]
                # lorafilter
                results += [gr.update(value="")]
                # lora_gallery
                results += [update_lora_filter("")]
                # style_selection
                results += [gr.update(choices=list(load_styles().keys()))]
                # mm_filter
                results += [gr.update(value="")]
                # mm_gallery
                results += [update_mm_filter("")]

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

            @performance_selection.change(
                inputs=[performance_selection],
                outputs=[perf_name]
                + performance_outputs
            )
            def performance_changed(selection):
                if selection == performance_settings.CUSTOM_PERFORMANCE:
                    return (
                        [perf_name.update(value="")]
                        + [gr.update(visible=True)] * len(performance_outputs)
                    )
                else:
                    return (
                        [perf_name.update(visible=False)]
                        + [gr.update(visible=False)] * len(performance_outputs)
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
