# https://github.com/twri/sdxl_prompt_styler/blob/main/sdxl_styles.json
import os
import shutil
from csv import DictReader, reader

styles = []

# Check if styles.csv file exists
if not os.path.isfile("styles.csv"):
    shutil.copy("styles.default", "styles.csv")

with open("styles.csv", "r") as file:
    # Create DictReader object to read CSV as dictionaries
    treader = DictReader(file)
    # Create default styles list
    default_styles = [
        {"name": "None", "prompt": "{prompt}", "negative_prompt": ""},
    ]
    # Read in styles from CSV into styles list
    styles = list(treader)
    # Insert default styles at start of styles list
    for item in default_styles:
        styles.insert(0, item)


styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in styles}
default_style = styles["None"]
style_keys = list(styles.keys())

SD_XL_BASE_RATIOS = {}

# Check if resolutions.csv file exists
if not os.path.isfile("resolutions.csv"):
    shutil.copy("resolutions.default", "resolutions.csv")

with open("resolutions.csv", "r") as file:
    reader = DictReader(file)
    for row in reader:
        SD_XL_BASE_RATIOS[row["ratio"]] = (int(row["width"]), int(row["height"]))


aspect_ratios = {f"{v[0]}x{v[1]} ({k})": v for k, v in SD_XL_BASE_RATIOS.items()}


def apply_style(style, positive, negative):
    prompt = ""
    negative_prompt = ""
    if not style:
        return positive, negative
    for item in style:
        p, n = styles.get(item, default_style)
        prompt += p + ", "
        negative_prompt += n + ", "

    return prompt.replace("{prompt}", positive), negative_prompt + ", " + negative
