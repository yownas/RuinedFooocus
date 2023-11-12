import os
import re
import random
import json

from modules.sdxl_styles import apply_style, allstyles
from random_prompt.build_dynamic_prompt import build_dynamic_prompt


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
    placeholders = re.findall(r"__([\w:]+(?:[\w\s]+)?)__", wildcard_text)
    placeholder_choices = {}  # Store random choices for each placeholder
    official_directory = "wildcards_official"
    directories = []
    directories.append(directory)
    directories.append(official_directory)


    for placeholder in placeholders:

        # Some one button prompt specials
        if placeholder.startswith("onebutton"):
            subjectoverride = ""
            placeholdersplit = placeholder.split(":",1)
            if len(placeholdersplit) > 1:
                subjectoverride = placeholdersplit[1]

            insertprompt = []
            if placeholder.startswith("onebuttonprompt"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=5, givensubject=subjectoverride))
            elif placeholder.startswith("onebuttonsubject"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride))
            elif placeholder.startswith("onebuttonhumanoid"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="humanoid"))
            elif placeholder.startswith("onebuttonmale"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="humanoid", gender = "male"))
            elif placeholder.startswith("onebuttonfemale"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="humanoid", gender = "female"))
            elif placeholder.startswith("onebuttonanimal"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="animal"))
            elif placeholder.startswith("onebuttonobject"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="object"))
            elif placeholder.startswith("onebuttonlandscape"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="landscape"))
            elif placeholder.startswith("onebuttonconcept"):
                insertprompt.append(build_dynamic_prompt(insanitylevel=7, imagetype="subject only mode", givensubject=subjectoverride, forcesubject="concept"))
            placeholder_choices[placeholder] = insertprompt

            
        elif placeholder not in placeholder_choices:
            found = False
            for dir in directories:
                for root, dirs, files in os.walk(dir):
                    if f"{placeholder}.txt" in files:
                        file_path = os.path.join(root, f"{placeholder}.txt")
                        with open(file_path, encoding="utf-8") as f:
                            words = f.read().splitlines()
                        placeholder_choices[placeholder] = words
                        found = True
                        break
                if found == True:
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
    while "Style: Pick Random" in style:
        style[style.index("Style: Pick Random")] = random.choice(allstyles)
    
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
