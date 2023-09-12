import os
import re
import random

from modules.sdxl_styles import apply_style


def process_wildcards(wildcard_text, directory="wildcards"):
    placeholders = re.findall(r"__(\w+)__", wildcard_text)
    for placeholder in placeholders:
        try:
            with open(os.path.join(directory, f"{placeholder}.txt")) as f:
                words = f.read().splitlines()
            wildcard_text = re.sub(rf"__{placeholder}__", random.choice(words), wildcard_text)
        except IOError:
            print(f"Error: Could not open file {placeholder}.txt. Please ensure the file exists and is readable.")
            raise
    return wildcard_text


def process_prompt(style, prompt, negative):
    p_txt, n_txt = apply_style(style, prompt, negative)
    p_txt = process_wildcards(p_txt)
    return p_txt, n_txt
