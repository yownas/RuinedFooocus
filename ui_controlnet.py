import modules.controlnet as controlnet
from modules.controlnet import (
    cn_options,
    load_cnsettings,
    save_cnsettings,
    NEWCN,
)

import gradio as gr
from shared import add_ctrl


def add_controlnet_tab():
    with gr.Tab(label="PowerUp"):
        with gr.Row():
            cn_selection = gr.Dropdown(
                label="Cheat Code",
                choices=["None"] + list(cn_options.keys()) + [NEWCN],
                value="None",
            )
            add_ctrl("cn_selection", cn_selection)

        cn_name = gr.Textbox(
            show_label=False,
            placeholder="Name",
            interactive=True,
            visible=False,
        )
        cn_save_btn = gr.Button(
            value="Save",
            visible=False,
        )

        cn_type = gr.Dropdown(
            label="Type",
            choices=map(lambda x: x.capitalize(), controlnet.controlnet_models.keys()),
            value=list(controlnet.controlnet_models.keys())[0].capitalize(),
            visible=False,
        )
        add_ctrl("cn_type", cn_type)

        cn_edge_low = gr.Slider(
            label="Edge (low)",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.2,
            visible=False,
        )
        add_ctrl("cn_edge_low", cn_edge_low)

        cn_edge_high = gr.Slider(
            label="Edge (high)",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.8,
            visible=False,
        )
        add_ctrl("cn_edge_high", cn_edge_high)

        cn_start = gr.Slider(
            label="Start",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            visible=False,
        )
        add_ctrl("cn_start", cn_start)

        cn_stop = gr.Slider(
            label="Stop",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=1.0,
            visible=False,
        )
        add_ctrl("cn_stop", cn_stop)

        cn_strength = gr.Slider(
            label="Strength",
            minimum=0.0,
            maximum=2.0,
            step=0.01,
            value=1.0,
            visible=False,
        )
        add_ctrl("cn_strength", cn_strength)

        cn_outputs = [
            cn_name,
            cn_save_btn,
            cn_type,
            cn_start,
            cn_stop,
            cn_strength,
        ]
        cn_preprocess_outputs = [
            cn_edge_low,
            cn_edge_high,
        ]

        @cn_selection.change(
            inputs=[cn_selection],
            outputs=[cn_name] + cn_outputs + cn_preprocess_outputs,
        )
        def cn_changed(selection):
            if selection != NEWCN:
                return [gr.update(visible=False)] + [gr.update(visible=False)] * len(
                    cn_outputs + cn_preprocess_outputs
                )
            else:
                return [gr.update(value="")] + [gr.update(visible=True)] * len(
                    cn_outputs + cn_preprocess_outputs
                )

        @cn_type.change(
            inputs=[cn_type],
            outputs=cn_preprocess_outputs,
        )
        def cn_type_changed(selection):
            if selection.lower() == "canny":
                return [gr.update(visible=True)] * len(cn_preprocess_outputs)
            else:
                return [gr.update(visible=False)] * len(cn_preprocess_outputs)

        @cn_save_btn.click(
            inputs=cn_outputs + cn_preprocess_outputs,
            outputs=[cn_selection],
        )
        def cn_save(
            cn_name,
            cn_save_btn,
            cn_type,
            cn_start,
            cn_stop,
            cn_strength,
            cn_edge_low,
            cn_edge_high,
        ):
            if cn_name != "":
                cn_options = load_cnsettings()
                opts = {
                    "type": cn_type.lower(),
                    "start": cn_start,
                    "stop": cn_stop,
                    "strength": cn_strength,
                }
                if cn_type.lower() == "canny":
                    opts.update(
                        {
                            "edge_low": cn_edge_low,
                            "edge_high": cn_edge_high,
                        }
                    )
                cn_options[cn_name] = opts
                save_cnsettings(cn_options)
                choices = list(cn_options.keys()) + [NEWCN]
                return gr.update(choices=choices, value=cn_name)
            else:
                return gr.update()

        input_image = gr.Image(
            label="Input image",
            type="pil",
            visible=True,
        )
        add_ctrl("input_image", input_image)
