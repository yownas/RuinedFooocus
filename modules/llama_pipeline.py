import re
from llama_cpp import Llama
from modules.util import TimeIt
from pathlib import Path
from modules.settings import default_settings

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

        repo = default_settings.get("llama_repo", "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF")
        file = default_settings.get("llama_file", "*q8_0.gguf")

        print(f"Loading {repo}")
        llm = Llama.from_pretrained(repo_id=repo , filename=file, verbose=False, n_ctx=4096)

        with TimeIt(""):
            print(f"# System:\n{system_prompt.strip()}\n")
            print(f"# User:\n{prompt.strip()}\n")
            print(f"# {name}: (Thinking...)")
            try:
                res = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ])["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"LLAMA ERROR: {e}")
                res = prompt

            print(f"{res.strip()}\n")

        llm._stack.close()
        llm.close()

        return res