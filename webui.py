import gradio as gr
import random
import time
import shared
import argparse
import modules.path
import fooocus_version
import modules.html
import modules.async_worker as worker

from modules.sdxl_styles import style_keys, aspect_ratios, styles
from modules.settings import load_settings

import ui_onebutton


def load_images_handler(files):
    return list(map(lambda x: x.name, files))


def generate_clicked(*args):
    yield gr.update(interactive=False), gr.update(
        visible=True, value=modules.html.make_progress_html(1, "Processing text encoding ...")
    ), gr.update(visible=True, value=None), gr.update(visible=False)

    worker.buffer.append(list(args))
    finished = False

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == "preview":
                percentage, title, image = product
                yield gr.update(interactive=False), gr.update(
                    visible=True, value=modules.html.make_progress_html(percentage, title)
                ), gr.update(visible=True, value=image) if image is not None else gr.update(), gr.update(visible=False)
            if flag == "results":
                yield gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False), gr.update(
                    visible=True, value=product
                )
                finished = True
    return


<<<<<<< HEAD
=======
def metadata_to_ctrls(metadata, ctrls):
    if not isinstance(metadata, Mapping):
        return ctrls

    if 'prompt' in metadata:
        ctrls[0] = metadata['prompt']
    if 'negative_prompt' in metadata:
        ctrls[1] = metadata['negative_prompt']
    if 'style' in metadata:
        ctrls[2] = metadata['style']
    if 'performance' in metadata:
        ctrls[3] = metadata['performance']
    if 'width' in metadata and 'height' in metadata:
        ctrls[4] = get_resolution_string(metadata['width'], metadata['height'])
    elif 'resolution' in metadata:
        ctrls[4] = metadata['resolution']
    # image_number
    if 'seed' in metadata:
        ctrls[6] = metadata['seed']
        ctrls[32] = False
    if 'sharpness' in metadata:
        ctrls[7] = metadata['sharpness']
    if 'sampler_name' in metadata:
        ctrls[8] = metadata['sampler_name']
    elif 'sampler' in metadata:
        ctrls[8] = metadata['sampler']
    if 'scheduler' in metadata:
        ctrls[9] = metadata['scheduler']
    if 'steps' in metadata:
        ctrls[10] = metadata['steps']
        if ctrls[10] == constants.STEPS_SPEED:
            ctrls[3] = 'Speed'
        elif ctrls[10] == constants.STEPS_QUALITY:
            ctrls[3] = 'Quality'
        else:
            ctrls[3] = 'Custom'
    if 'switch' in metadata:
        ctrls[11] = round(metadata['switch'] / ctrls[10], 2)
        if ctrls[11] != round(constants.SWITCH_SPEED / constants.STEPS_SPEED, 2):
            ctrls[3] = 'Custom'
    if 'cfg' in metadata:
        ctrls[12] = metadata['cfg']
    if 'base_model' in metadata:
        ctrls[13] = metadata['base_model']
    elif 'base_model_name' in metadata:
        ctrls[13] = metadata['base_model_name']
    if 'refiner_model' in metadata:
        ctrls[14] = metadata['refiner_model']
    elif 'refiner_model_name' in metadata:
        ctrls[14] = metadata['refiner_model_name']
    if 'base_clip_skip' in metadata:
        ctrls[15] = metadata['base_clip_skip']
    if 'refiner_clip_skip' in metadata:
        ctrls[16] = metadata['refiner_clip_skip']
    if 'l1' in metadata:
        ctrls[17] = metadata['l1']
    if 'w1' in metadata:
        ctrls[18] = metadata['w1']
    if 'l2' in metadata:
        ctrls[19] = metadata['l2']
    if 'w2' in metadata:
        ctrls[20] = metadata['w2']
    if 'l3' in metadata:
        ctrls[21] = metadata['l3']
    if 'w3' in metadata:
        ctrls[22] = metadata['w3']
    if 'l4' in metadata:
        ctrls[23] = metadata['l4']
    if 'w4' in metadata:
        ctrls[24] = metadata['w4']
    if 'l5' in metadata:
        ctrls[25] = metadata['l5']
    if 'w5' in metadata:
        ctrls[26] = metadata['w5']
    # save_metadata_json
    # save_metadata_png
    if 'img2img' in metadata:
        ctrls[29] = metadata['img2img']
        if 'start_step' in metadata:
            if ctrls[3] == 'Speed':
                ctrls[30] = round(metadata['start_step'] / constants.STEPS_SPEED, 2)
            elif ctrls[3] == 'Quality':
                ctrls[30] = round(metadata['start_step'] / constants.STEPS_QUALITY, 2)
            else:
                ctrls[30] = round(metadata['start_step'] / ctrls[10], 2)
        if 'denoise' in metadata:
            ctrls[31] = metadata['denoise']
    # seed_random
    return ctrls    


def load_prompt_handler(_file, *args):
    ctrls=list(args)
    path = _file.name
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as json_file:
            try:
                json_obj = json.load(json_file)
                metadata_to_ctrls(json_obj, ctrls)
            except Exception as e:
                print(e)
            finally:
                json_file.close()
    elif path.endswith('.png'):
        with open(path, 'rb') as png_file:
            image = Image.open(png_file)
            png_file.close()
            if 'Comment' in image.info:
                try:
                    metadata = json.loads(image.info['Comment'])
                    metadata_to_ctrls(metadata, ctrls)
                except Exception as e:
                    print(e)
    return ctrls


def load_images_handler(files):
    return list(map(lambda x: x.name, files))


>>>>>>> 3e14506 (Fixed loading denoise value, re-ordered buttons)
settings = load_settings()

if settings["theme"] == "None":
    theme = gr.themes.Default()
else:
    theme = settings["theme"]

shared.gradio_root = gr.Blocks(
    theme=theme, title="RuinedFooocus " + fooocus_version.version, css=modules.html.css
).queue()
with shared.gradio_root:
    with gr.Row():
        with gr.Column():
            progress_window = gr.Image(label="Preview", show_label=True, height=640, visible=False)
            progress_html = gr.HTML(
                value=modules.html.make_progress_html(32, "Progress 32%"),
                visible=False,
                elem_id="progress-bar",
                elem_classes="progress-bar",
            )
            gallery = gr.Gallery(label="Gallery", show_label=False, object_fit="contain", height=720, visible=True)
            with gr.Row(elem_classes="type_row"):
                with gr.Column(scale=0.85):
                    prompt = gr.Textbox(
                        show_label=False,
                        placeholder="Type prompt here.",
                        container=False,
                        autofocus=True,
                        elem_classes="type_row",
                        lines=1024,
                        value=settings["prompt"],
                    )
                with gr.Column(scale=0.15, min_width=0):
<<<<<<< HEAD
                    run_button = gr.Button(label="Generate", value="Generate", elem_classes="type_row")
=======
                    with gr.Row():
                        load_images_button = gr.UploadButton(label='Load Image(s)', file_count='multiple', file_types=["image"], elem_classes='type_small_row')
                    with gr.Row():
                        load_prompt_button = gr.UploadButton(label='Load Prompt', file_count='single', file_types=['.json', '.png'], elem_classes='type_small_row')
                with gr.Column(scale=0.15, min_width=0):
                    with gr.Row():
                        img2img_mode = gr.Checkbox(label='Image-2-Image', value=settings['img2img_mode'], elem_classes='type_small_row')
                    with gr.Row():
                        run_button = gr.Button(label='Generate', value='Generate', elem_classes='type_small_row')
>>>>>>> 3e14506 (Fixed loading denoise value, re-ordered buttons)
            with gr.Row():
                advanced_checkbox = gr.Checkbox(label="Advanced", value=settings["advanced_mode"], container=False)
        with gr.Column(scale=0.5, visible=settings["advanced_mode"]) as right_col:
            with gr.Tab(label="Setting"):
                performance_selction = gr.Radio(
                    label="Performance", choices=["Speed", "Quality"], value=settings["performance"]
                )
                aspect_ratios_selction = gr.Dropdown(
                    label="Aspect Ratios (width x height)",
                    choices=list(aspect_ratios.keys()),
                    value=settings["resolution"],
                )
                style_selction = gr.Dropdown(
                    label="Style Selection",
                    multiselect=True,
                    container=True,
                    choices=style_keys,
                    value=settings["style"],
                )
                style_button = gr.Button(value="⬅️ Send Style to prompt", size="sm")
                image_number = gr.Slider(
                    label="Image Number", minimum=1, maximum=50, step=1, value=settings["image_number"]
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    show_label=True,
                    placeholder="Type prompt here.",
                    value=settings["negative_prompt"],
                )
                seed_random = gr.Checkbox(label="Random Seed", value=settings["seed_random"])
                image_seed = gr.Number(
                    label="Seed", value=settings["seed"], precision=0, visible=not settings["seed_random"]
                )

                with gr.Row():
                    img2img_mode = gr.Checkbox(
                        label="Image 2 Image", value=settings["img2img_mode"], elem_classes="type_small_row"
                    )
                    load_images_button = gr.UploadButton(
                        label="Load Image(s)",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes="type_small_row",
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
                    apply_style, inputs=[prompt, style_selction], outputs=[prompt, negative_prompt, style_selction]
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
                                label="Strength", minimum=-2, maximum=2, step=0.01, value=settings[f"lora_{i+1}_weight"]
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
                img2img_start_step = gr.Slider(
                    label="Image 2 Image Start Step",
                    minimum=0.0,
                    maximum=0.5,
                    step=0.01,
                    value=settings["img2img_start_step"],
                )
                img2img_denoise = gr.Slider(
                    label="Image 2 Image Denoise",
                    minimum=0.5,
                    maximum=1.0,
                    step=0.01,
                    value=settings["img2img_denoise"],
                )
                sharpness = gr.Slider(
                    label="Sampling Sharpness", minimum=0.0, maximum=40.0, step=0.01, value=settings["sharpness"]
                )
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117">\U0001F4D4 Document</a>')

            def model_refresh_clicked():
                modules.path.update_all_model_names()
                results = []
                results += [
                    gr.update(choices=modules.path.model_filenames),
                    gr.update(choices=["None"] + modules.path.model_filenames),
                ]
                for i in range(5):
                    results += [gr.update(choices=["None"] + modules.path.lora_filenames), gr.update()]
                return results

            model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls)

            ui_onebutton.ui_onebutton(prompt)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, right_col)
        ctrls = [
            prompt,
            negative_prompt,
            style_selction,
            performance_selction,
            aspect_ratios_selction,
            image_number,
            image_seed,
            sharpness,
            save_metadata,
        ]

        img2imgcontrols = [img2img_mode, img2img_start_step, img2img_denoise]
        load_images_button.upload(fn=load_images_handler, inputs=[load_images_button], outputs=gallery)
        ctrls += [base_model, refiner_model] + lora_ctrls + img2imgcontrols
        run_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed).then(
            fn=generate_clicked, inputs=ctrls + [gallery], outputs=[run_button, progress_html, progress_window, gallery]
        )


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
parser.add_argument("--share", action="store_true", help="Set whether to share on Gradio.")
parser.add_argument(
    "--listen", type=str, default=None, metavar="IP", nargs="?", const="0.0.0.0", help="Set the listen interface."
)
args = parser.parse_args()
favicon_path = "logo.ico"
shared.gradio_root.launch(
    inbrowser=True,
    server_name=args.listen,
    server_port=args.port,
    share=args.share,
    favicon_path=favicon_path,
)
