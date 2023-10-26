import os
import re
import random
import json

from modules.sdxl_styles import apply_style


def process_metadata(gen_data):
    try:
        meta = json.loads(gen_data["prompt"])
        meta = dict((k.lower(), v) for k, v in meta.items())
        gen_data.update(meta)
        if "prompt" in meta:
            gen_data["style_selection"] = None

        if "loras" in meta:
            idx = 1
            for lora in re.findall(r"<(.*?):(.*?)>", meta["loras"]):
                l, w = lora
                gen_data[f"l{idx}"] = l
                gen_data[f"w{idx}"] = float(w)
                idx += 1
    except:
        pass
    return gen_data


def get_promptlist(gen_data):
    return gen_data["prompt"].split("---")


def process_wildcards(wildcard_text, directory="wildcards"):
    placeholders = re.findall(r"__(\w+)__", wildcard_text)
    placeholder_choices = {}  # Store random choices for each placeholder

    for placeholder in placeholders:
        if placeholder not in placeholder_choices:
            found = False
            for root, dirs, files in os.walk(directory):
                if f"{placeholder}.txt" in files:
                    file_path = os.path.join(root, f"{placeholder}.txt")
                    with open(file_path, encoding="utf-8") as f:
                        words = f.read().splitlines()
                    placeholder_choices[placeholder] = words
                    found = True
                    break

            if not found:
                placeholder_choices[placeholder] = [placeholder]
                print(
                    f"Error: Could not find file {placeholder}.txt in {directory} or its subdirectories."
                )

    for placeholder in placeholders:
        random_choice = random.choice(placeholder_choices[placeholder])
        wildcard_text = re.sub(
            rf"__{placeholder}__", random_choice, wildcard_text, count=1
        )

    return wildcard_text


def process_prompt(style, prompt, negative):
    pattern = re.compile(r"<style:([^>]+)>")
    styles = [] if style is None else style.copy()
    for match in re.finditer(pattern, prompt):
        styles += [f"Style: {match.group(1)}"]
    prompt = re.sub(pattern, "", prompt)
    p_txt, n_txt = apply_style(styles, prompt, negative)
    p_txt = process_wildcards(p_txt)
    return p_txt, n_txt


def parse_loras(prompt, negative):
    pattern = re.compile(r"<lora:([^>]+):(\d*\.*\d+)>")
    loras = []
    for match in re.finditer(pattern, prompt):
        loras.append((f"{match.group(1)}.safetensors", float(match.group(2))))
    for match in re.finditer(pattern, negative):
        loras.append((f"{match.group(1)}.safetensors", float(match.group(2))))
    return loras, re.sub(pattern, "", prompt), re.sub(pattern, "", negative)
