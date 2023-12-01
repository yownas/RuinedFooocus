import gradio as gr
from transformers import CLIPTokenizer
from shared import add_ctrl
import random

# FIXME use global tokenizer?
#global tokenizer
#evolve_tokenizer = tokenizer

def add_evolve_tab(prompt, run_event):
    evolve_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with gr.Tab(label="Evo"):
        with gr.Row():
            evolve_1_btn = gr.Button(
                value="1",
                min_width=1,
            )
            evolve_2_btn = gr.Button(
                value="2",
                min_width=1,
            )
            evolve_3_btn = gr.Button(
                value="3",
                min_width=1,
            )
        with gr.Row():
            evolve_4_btn = gr.Button(
                value="4",
                min_width=1,
            )
            evolve_5_btn = gr.Button(
                value="5",
                min_width=1,
            )
            evolve_6_btn = gr.Button(
                value="6",
                min_width=1,
            )
        with gr.Row():
            evolve_7_btn = gr.Button(
                value="7",
                min_width=1,
            )
            evolve_8_btn = gr.Button(
                value="8",
                min_width=1,
            )
            evolve_9_btn = gr.Button(
                value="9",
                min_width=1,
            )
        with gr.Row():
            evolve_strength = gr.Slider(
                minimum=0,
                maximum=100,
                value=10,
                step=1,
                label="Evolve chance %:"
            )
        with gr.Row():
            evo_help = gr.HTML(value='''
                Start with any prompt.<br>
                For best result, set a non-random seed.<br>
                Click on the number the correspond to the image you like best.
            ''')

        def tokenize_and_randomize(prompt, strength):
            all_tokens = list(evolve_tokenizer.get_vocab().keys())
            tokens = evolve_tokenizer.tokenize(prompt)
            res = []
            for token in tokens:
                if random.random() < float(strength / 100.0):
                    res += [all_tokens[random.randint(0, len(all_tokens) - 3)]] # Skip <|startoftext> & <|endoftext|>
                else:
                    res += [token]
            return evolve_tokenizer.convert_tokens_to_string(res)

        def four_evolved_prompts(prompt, strength):
            res = []
            for i in range(4):
                res.append(tokenize_and_randomize(prompt, strength))
            return res

        def evolve(
            button,
            strength,
            prompt,
            run_event,
        ):
            prompts = prompt.split("---")
            in_txt = prompts[min(int(button), len(prompts)) - 1]
            res = four_evolved_prompts(in_txt, strength) + [in_txt] + four_evolved_prompts(in_txt, strength)
            return gr.update(value='\n---\n'.join(res)), run_event+1

        evolve_1_btn.click(evolve, inputs=[evolve_1_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_2_btn.click(evolve, inputs=[evolve_2_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_3_btn.click(evolve, inputs=[evolve_3_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_4_btn.click(evolve, inputs=[evolve_4_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_5_btn.click(evolve, inputs=[evolve_5_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_6_btn.click(evolve, inputs=[evolve_6_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_7_btn.click(evolve, inputs=[evolve_7_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_8_btn.click(evolve, inputs=[evolve_8_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
        evolve_9_btn.click(evolve, inputs=[evolve_9_btn, evolve_strength, prompt, run_event], outputs=[prompt, run_event])
