import re
try:
    from nexa.gguf.llama.llama import Llama
except:
    print("ERROR: Could not load Llama.")
    Llama = None
from txtai import Embeddings
from modules.util import TimeIt
from pathlib import Path
from modules.util import url_to_filename, load_file_from_url
from shared import path_manager, settings, local_url
import json
import xmltodict
import modules.async_worker as worker

def llama_names():
        names = []
        folder_path = Path("llamas")
        for path in folder_path.rglob("*"):
            if path.suffix.lower() in [".txt"]:
                f = open(path, "r", encoding='utf-8')
                name = f.readline().strip()
                names.append((name, str(path)))
        names.sort(key=lambda x: x[0].casefold())
        return names

def run_llama(system_file, prompt):
        if Llama == None:
            return "Error: There is no Llama"
        name = None
        sys_pat = r"system:.*\n\n"
        system = re.match(sys_pat, prompt, flags=re.M|re.I)
        if system is not None: # Llama system-prompt provided in the ui-prompt
            name = "Llama"
            system_prompt = re.sub(r"^[^:]*: *", "", system.group(0), flags=re.M|re.I)
            prompt = re.sub(sys_pat, "", prompt)
        else:
            try:
                file = open(system_file, "r", encoding='utf-8')
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
                ret = llama.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    repeat_penalty = 1.18,
                )["choices"][0]["message"]
                res = ret["content"]
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
    embeddings_hash = ""

    def parse_gen_data(self, gen_data):
        return gen_data

    def load_base_model(self):
        localfile = settings.default_settings.get("llama_localfile", None)
        repo = settings.default_settings.get("llama_repo", "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF")
        file = settings.default_settings.get("llama_file", "*q8_0.gguf")
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
                    model_dir="cache/embeds",
                    progress=True,
                    file_name=url_to_filename(source[1]),
                )
                file = open(filename, "r", encoding='utf-8')
                data = file.read()
                file.close()

                if source[1].endswith(".md"):
                    data = data.split("\n#")
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
        if Llama == None:
            return "Error: There is no Llama"

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
        if self.embeddings_hash != str(embed):
            self.embeddings_hash = str(embed)
            self.embeddings = None
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
            for data in self.embeddings.search(q, limit=3):
                #if data["score"] >= 0.5:
                context += data["text"] + "\n\n"
            system_prompt += context

        if settings.default_settings.get("enable_llm_tools", False):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": "Generates an image from a prompt.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "The prompt for the image"},
                            },
                            "required": ["prompt"],
                        },
                    },
                },
            ]
            tool_prompt = "\nUse the tool when asked to generate an image. When using the tool you must make sure you use the correct format.\n"
        else:
            tools = None
            tool_prompt = ""
        chat = [{"role": "system", "content": system_prompt + tool_prompt}] + h[-3 if len(h) > 3 else -len(h):] # Keep just the last 3 messages

        print(f"Thinking...")
        with TimeIt("LLM thinking"):
            response = self.llm.create_chat_completion(
                messages = chat,
                max_tokens=1024,
                stream=True,
                tools=tools,
                tool_choice="auto",
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

            call = None
            if settings.default_settings.get("enable_llm_tools", False):
                # Parse the output and look for tool_calls. (This is probably not the proper way to do it...)
                tool_error = f"![Error](gradio_api/file=html/error.png)"
                if "<tool_call>" in text:
                    try:
                        call = re.match(r"^.*<tool_call>(?P<call>.+)</tool_call>.*$", text, flags=re.MULTILINE+re.DOTALL)
                        call = call.groupdict()['call'].strip()[1:-1] # There are extra {} in the reply?
                        call_type = "xml"
                    except:
                        call = None
                else:
                    # Llama-instruct?
                    try:
                        json.loads(text)
                        call = text
                        call_type = "json"
                    except Exception as e:
                        call = None

                if call is not None:
                    try:
                        tool_call = json.loads(call) # There are extra {} in the reply?
                        if tool_call.get('name', None) == 'generate_image':
                            if 'arguments' in tool_call:
                                prompt = tool_call['arguments']['prompt']
                            elif 'parameters' in tool_call:
                                if 'properties' in tool_call['parameters']:
                                    prompt = tool_call['parameters']['properties']['prompt']
                                else:
                                    prompt = tool_call['parameters']['prompt']
                            else:
                                # Unknown...
                                prompt = str(tool_call)

                            task_id = -1

                            tmp_data = {
                                'task_type': "tool_call",
                                'task_id': task_id,
                                'silent': True,
                                'prompt': prompt,
                                'negative': "",
                                'loras': [
                                    ("", f"{settings.default_settings.get('lora_1_weight', 1.0)} - {settings.default_settings.get('lora_1_model', 'None')}"),
                                    ("", f"{settings.default_settings.get('lora_2_weight', 1.0)} - {settings.default_settings.get('lora_2_model', 'None')}"),
                                    ("", f"{settings.default_settings.get('lora_3_weight', 1.0)} - {settings.default_settings.get('lora_3_model', 'None')}"),
                                    ("", f"{settings.default_settings.get('lora_4_weight', 1.0)} - {settings.default_settings.get('lora_4_model', 'None')}"),
                                    ("", f"{settings.default_settings.get('lora_5_weight', 1.0)} - {settings.default_settings.get('lora_5_model', 'None')}"),
                                ],
                                'style_selection': settings.default_settings['style'],
                                'seed': -1,
                                'base_model_name': settings.default_settings['base_model'],
                                'performance_selection': settings.default_settings['performance'],
                                'aspect_ratios_selection': settings.default_settings["resolution"],
                                'cn_selection': None,
                                'cn_type': None,
                                'silent': True,
                                'image_number': 1,
                            }

                            # unload llm model from memory?
                            # TODO: make this selectable for people with more ram/vram that is socialy acceptable
                            del self.llm
                            self.llm = None

                            info_txt = "(Generating image...)"
                            if call_type == "xml":
                                tmp_text = re.sub(r"<tool_call>.*</tool_call>", info_txt, text, flags=re.MULTILINE+re.DOTALL)
                            elif call_type == "json":
                                tmp_text = info_txt
                            else:
                                # Just add it to the end...
                                tmp_text = text + "\n" + info_txt

                            worker.add_result(
                                gen_data["task_id"],
                                "preview",
                                gen_data["history"] + [{"role": "assistant", "content": tmp_text}]
                            )

                            results = worker._process(tmp_data.copy())
                            file = results[0]
                            filename = str(file.relative_to(file.cwd()).as_posix())
                            url = "gradio_api/file=" + re.sub(r'[^/]+/\.\./', '', filename)
                            markdown = f"\n*{prompt}*\n\n![Image]({url})\n"

                            if call_type == "xml":
                                text = re.sub(r"<tool_call>.*</tool_call>", markdown, text, flags=re.MULTILINE+re.DOTALL)
                            elif call_type == "json":
                                text = markdown
                            else:
                                # Just add it to the end...
                                text += "\n" + markdown

                        else:
                            text = tool_error # Unknown tool
                            text += "Unknown tool"

                    except Exception as e:
                        import traceback
                        print(f"ERROR:")
                        traceback.print_exc()
                        text += f"Error: {e}\n\n"
                        text += f"Call: {call}\n\n"
                        text += "Looks like I made a mistake. I really need to make sure I use the correct format. Do you want me to try again?"

        return text
