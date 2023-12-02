import gradio as gr
from shared import tokenizer
import random

def add_evolve_tab(prompt, run_event):
    def tokenize_and_randomize(prompt, strength):
        all_tokens = list(tokenizer.get_vocab().keys())
        tokens = tokenizer.tokenize(prompt)
        res = []
        for token in tokens:
            if random.random() < float(strength / 100.0):
                res += [all_tokens[random.randint(0, len(all_tokens) - 3)]] # Skip <|startoftext> & <|endoftext|>
            else:
                res += [token]
        return tokenizer.convert_tokens_to_string(res)

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

    with gr.Tab(label="Evo"):
        evolve_btn = {}
        for x in range(0, 3):
            with gr.Row():
                for y in range(1, 4):
                    evolve_btn[3*x+y] = gr.Button(
                        value=str(3*x+y),
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
                Start with any prompt. Random chunk of letters works great.<br>
                Click on the number that correspond to the image you like best.<br>
                Repeat.<br>
                For best result, set a static seed.<br>
            ''')

        for i in range(1, 10):
            evolve_btn[i].click(evolve, inputs=[evolve_btn[i], evolve_strength, prompt, run_event], outputs=[prompt, run_event])