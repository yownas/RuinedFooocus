import os
import shutil
import json
import random
from csv import DictReader, reader

DEFAULT_STYLES_FILE = "settings/styles.default"
STYLES_FILE = "settings/styles.csv"
DEFAULT_RESOLUTIONS_FILE = "settings/resolutions.default"
RESOLUTIONS_FILE = "settings/resolutions.json"


def load_styles():
    styles = []

    if not os.path.isfile(STYLES_FILE):
        shutil.copy(DEFAULT_STYLES_FILE, STYLES_FILE)

    with open(STYLES_FILE, "r") as f:
        reader = DictReader(f)
        styles = list(reader)

    default_style = {"name": "None", "prompt": "{prompt}", "negative_prompt": ""}
    random_style = {"name": "Style: Pick Random", "prompt": "{prompt}", "negative_prompt": ""}
    styles.insert(0, random_style)
    styles.insert(0, default_style)

    return {s["name"]: (s["prompt"], s["negative_prompt"]) for s in styles}


def load_resolutions():
    ratios = {}

    if not os.path.isfile(RESOLUTIONS_FILE):
        shutil.copy(DEFAULT_RESOLUTIONS_FILE, RESOLUTIONS_FILE)

    with open(RESOLUTIONS_FILE) as f:
        data = json.load(f)
        for ratio, res in data.items():
            ratios[ratio] = (res["width"], res["height"])

    return ratios


def apply_style(style, prompt, negative_prompt):
    output_prompt = ""
    output_negative_prompt = ""

    
    while "Style: Pick Random" in style:
        style[style.index("Style: Pick Random")] = random.choice(allstyles)

    if not style:
        return prompt, negative_prompt

    for s in style:
        p, n = styles.get(s, default_style)
        output_prompt += p + ", "
        output_negative_prompt += n + ", "

    output_prompt = output_prompt.replace("{prompt}", prompt)
    output_negative_prompt += ", " + negative_prompt

    return output_prompt, output_negative_prompt


styles = load_styles()
default_style = styles["None"]
allstyles = [x for x in load_styles() if x.startswith('Style')]
allstyles.remove("Style: Pick Random")


SD_XL_BASE_RATIOS = load_resolutions()
aspect_ratios = {f"{v[0]}x{v[1]} ({k})": v for k, v in SD_XL_BASE_RATIOS.items()}
