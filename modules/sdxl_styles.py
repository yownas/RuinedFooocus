import random
import shutil
from csv import DictReader
from pathlib import Path

DEFAULT_STYLES_FILE = Path("settings/styles.default")
STYLES_FILE = Path("settings/styles.csv")


def load_styles():
    styles = []

    if not STYLES_FILE.is_file():
        shutil.copy(DEFAULT_STYLES_FILE, STYLES_FILE)

    with STYLES_FILE.open("r") as f:
        reader = DictReader(f)
        styles = list(reader)

    default_style = {"name": "None", "prompt": "{prompt}", "negative_prompt": ""}
    random_style = {
        "name": "Style: Pick Random",
        "prompt": "{prompt}",
        "negative_prompt": "",
    }
    styles.insert(0, random_style)
    styles.insert(0, default_style)

    return {s["name"]: (s["prompt"], s["negative_prompt"]) for s in styles}


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
allstyles = [x for x in load_styles() if x.startswith("Style")]
allstyles.remove("Style: Pick Random")
