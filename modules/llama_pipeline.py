import re
from llama_cpp import Llama
from txtai import Embeddings
from modules.util import TimeIt
from pathlib import Path
from modules.settings import default_settings
from modules.util import url_to_filename, load_file_from_url
from shared import path_manager
import modules.async_worker as worker
import json
import requests

def llama_names():
        names = []
        folder_path = Path("llamas")
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in [".txt"]:
                f = open(path, "r")
                name = f.readline().strip()
                names.append((name, str(path)))
        names.sort(key=lambda x: x[0].casefold())
        return names

def run_llama(system_file, prompt):
        name = None
        sys_pat = "system:.*\n\n"
        system = re.match(sys_pat, prompt, flags=re.M|re.I)
        if system is not None: # Llama system-prompt provided in the ui-prompt
            name = "Llama"
            system_prompt = re.sub("^[^:]*: *", "", system.group(0), flags=re.M|re.I)
            prompt = re.sub(sys_pat, "", prompt)
        else:
            try:
                file = open(system_file, "r")
                name = name if name is not None else file.readline().strip()
                system_prompt = file.read().strip()
            except:
                print(f"LLAMA ERROR: Could not open file {system_file}")
                return prompt

        llama = pipeline()
        llama.load_base_model()

        with TimeIt(""):
            print(f"# System:\n{system_prompt.strip()}\n")
            print(f"# User:\n{prompt.strip()}\n")
            print(f"# {name}: (Thinking...)")
            try:
                res = llama.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    repeat_penalty = 1.18,
                )["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"LLAMA ERROR: {e}")
                res = prompt

            print(f"{res.strip()}\n")

        llama.llm._stack.close()
        llama.llm.close()

        return res

class pipeline:
    pipeline_type = ["llama"]

    llm = None
    embeddings = None

    def parse_gen_data(self, gen_data):
        return gen_data

    def load_base_model(self):
        localfile = default_settings.get("llama_localfile", None)
        repo = default_settings.get("llama_repo", "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF")
        file = default_settings.get("llama_file", "*q8_0.gguf")
        with TimeIt("Load LLM"):
            if localfile is None:
                print(f"Loading {repo}")
                self.llm = Llama.from_pretrained(
                    repo_id=repo,
                    filename=file,
                    verbose=False,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    offload_kqv=True,
                    flash_attn=True,
                )
            else:
                llm_path = path_manager.get_folder_file_path(
                    "llm",
                    localfile,
                    default = Path(path_manager.model_paths["llm_path"]) / localfile
                )
                print(f"Loading {localfile}")
                self.llm = Llama(
                    model_path=str(llm_path),
                    verbose=False,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    offload_kqv=True,
                    flash_attn=True,
                )
        self.embeddings = None

    def index_source(self, source):
        if self.embeddings == None:
            self.embeddings = Embeddings(content=True)
            self.embeddings.initindex(reindex=True)

        match source[0]:

            case "url":
                print(f"Read {source[1]}")
                filename = load_file_from_url(
                    source[1],
                    model_dir="../cache/embeds",
                    file_name=url_to_filename(source[1]),
                )
                file = open(filename, "r")
                data = file.read()
                file.close()

                if source[1].endswith(".md"):
                    data = data.split("\n# ")
                elif source[1].endswith(".txt"):
                    data = data.split("\n\n")

            case "text":
                data = source[1]

            case _:
                print("WARNING: Unknown embedding type {source[0]}")
                return

        if data:
            self.embeddings.upsert(data)


    def process(self, gen_data):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            gen_data["history"]
        )

        if self.llm == None:
            self.load_base_model()

        # load embeds?
        # FIXME should dump the entire gen_data["embed"] to index_source() and have it sort it out
        embed = json.loads(gen_data['embed'])
        if embed:
            if not self.embeddings: # If chatbot has embeddings to index, check that we have them.
                for source in embed:
                    self.index_source(source)
        else:
            self.embeddings = None

        system_prompt = gen_data["system"]

        h = gen_data["history"]

        if self.embeddings:
            q = h[-1]["content"]
            context = "This some context that will help you answer the question:\n"
            for data in self.embeddings.search(q, limit=2):
                #if data["score"] >= 0.5:
                context += data["text"] + "\n\n"
            system_prompt += context

        chat = [{"role": "system", "content": system_prompt}] + h[-3 if len(h) > 3 else -len(h):] # Keep just the last 3 messages

        print(f"Thinking...")
        with TimeIt("LLM thinking"):
            response = self.llm.create_chat_completion(
                messages = chat,
                max_tokens=1024,
                stream=True,
            )
            #["choices"][0]["message"]["content"]

            text = ""
            for chunk in response:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    tokens = delta['content']
                    for token in tokens:
                        text += token
                        worker.add_result(
                            gen_data["task_id"],
                            "preview",
                            gen_data["history"] + [{"role": "assistant", "content": text}]
                        )

        return text
