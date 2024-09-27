import gradio as gr
import re
from llama_cpp import Llama

def add_llama_tab(prompt):
    def run_llama_run(prompt):
        llm = Llama.from_pretrained(repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF", filename="*q8_0.gguf")

        system_prompt = "You are an artist, describe a beautiful scene using the words from the user."

        sys_pat = "system:.*\n\n"
        system = re.match(sys_pat, prompt, flags=re.M|re.I)
        if system is not None:
            system_prompt = re.sub("system: *", "", system.group(0), flags=re.M|re.I)
            prompt = re.sub(sys_pat, "", prompt)

        res = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])["choices"][0]["message"]["content"]
        return gr.update(value=res)

    llama_btn = gr.Button(
        value="🦙",
        min_width=1,
    )
    llama_btn.click(run_llama_run, inputs=[prompt], outputs=[prompt])
