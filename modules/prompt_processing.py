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
            gen_data["style_selection"] = []

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
    placeholders = re.findall(r"__([\w:]+)__", wildcard_text)
    placeholders_onebutton = re.findall(r"__([\w]+:[^\s_]+(?:[^\s_]+|\s(?=[\w:]+))*)__", wildcard_text)
    placeholders += placeholders_onebutton
    placeholder_choices = {}  # Store random choices for each placeholder
    official_directory = "wildcards_official"
    directories = []
    directories.append(directory)
    directories.append(official_directory)

    for placeholder in placeholders:
        # Skip onebuttonprompt wildcards for now; handled below
        if placeholder.startswith("onebutton"):
            continue

        elif placeholder not in placeholder_choices:
            found = False
            for dir in directories:
                for root, dirs, files in os.walk(dir):
                    if f"{placeholder}.txt" in files:
                        file_path = os.path.join(root, f"{placeholder}.txt")
                        with open(file_path, encoding="utf-8") as f:
                            words = [word.strip() for word in f.read().splitlines() if not word.startswith("#")]
                        placeholder_choices[placeholder] = words
                        found = True
                        break
                if found == True:
                    break

            if not found:
                print(
                    f"Error: Could not find file {placeholder}.txt in {directory} or its subdirectories."
                )
                placeholder_choices[placeholder] = [f"{placeholder}"]

    for placeholder in placeholders:
        random_choice = ""

        # Some one button prompt specials
        if placeholder.startswith("onebutton"):
            subjectoverride = ""
            placeholdersplit = placeholder.split(":", 1)
            if len(placeholdersplit) > 1:
                subjectoverride = placeholdersplit[1]

            if placeholder.startswith("onebuttonprompt"):
                random_choice = build_dynamic_prompt(insanitylevel=5, givensubject=subjectoverride)
            elif placeholder.startswith("onebuttonsubject"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonhumanoid"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="humanoid",
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonmale"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="humanoid",
                        gender="male",
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonfemale"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="humanoid",
                        gender="female",
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonanimal"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="animal",
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonobject"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="object",
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonlandscape"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="landscape",
                        advancedprompting=False,
                    )
            elif placeholder.startswith("onebuttonconcept"):
                random_choice = build_dynamic_prompt(
                        insanitylevel=5,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        forcesubject="concept",
                        advancedprompting=False,
                    )
            #failover
            else:
                random_choice = build_dynamic_prompt(
                        insanitylevel=3,
                        imagetype="subject only mode",
                        givensubject=subjectoverride,
                        advancedprompting=False,
                    )

        # Regular wildcards
        else:
            random_choice = random.choice(placeholder_choices[placeholder])

        wildcard_text = re.sub(
            rf"__{placeholder}__", random_choice, wildcard_text, count=1
        )

    return wildcard_text


def process_prompt(style, prompt, negative, gen_data=[]):
    if(gen_data["obp_assume_direct_control"]):
                prompt = build_dynamic_prompt(
                    insanitylevel=gen_data["obp_insanitylevel"],
                    forcesubject=gen_data["obp_subject"],
                    artists=gen_data["obp_artist"],
                    subtypeobject=gen_data["obp_chosensubjectsubtypeobject"],
                    subtypehumanoid=gen_data["obp_chosensubjectsubtypehumanoid"],
                    subtypeconcept=gen_data["obp_chosensubjectsubtypeconcept"],
                    gender=gen_data["obp_chosengender"],
                    imagetype=gen_data["obp_imagetype"],
                    imagemodechance=gen_data["obp_imagemodechance"],
                    givensubject=gen_data["obp_givensubject"],
                    smartsubject=gen_data["obp_smartsubject"],
                    overrideoutfit=gen_data["obp_givenoutfit"],
                    prefixprompt=gen_data["obp_prefixprompt"],
                    suffixprompt=gen_data["obp_suffixprompt"],
                    giventypeofimage=gen_data["obp_giventypeofimage"],
                    antivalues=gen_data["obp_antistring"],
                    advancedprompting=False,
                )
    pattern = re.compile(r"<style:([^>]+)>")
    styles = [] if style is None else style.copy()
    for match in re.finditer(pattern, prompt):
        styles += [f"Style: {match.group(1)}"]
    prompt = re.sub(pattern, "", prompt)
    p_txt, n_txt = apply_style(styles, prompt, negative)
    wildcard_pattern = r"__([\w:]+)__"
    wildcard_pattern_onebutton = r"__([\w]+:[^\s_]+(?:[^\s_]+|\s(?=[\w:]+))*)__"
    while ((match := re.search(wildcard_pattern, p_txt)) or (match := re.search(wildcard_pattern_onebutton, p_txt))) is not None:
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
