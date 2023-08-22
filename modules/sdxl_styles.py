# https://github.com/twri/sdxl_prompt_styler/blob/main/sdxl_styles.json
import os
from csv import DictReader

styles = []

# Check if styles.csv file exists
if os.path.isfile("styles.csv"):
    # Open styles.csv file for reading
    with open("styles.csv", "r") as file:
        # Create DictReader object to read CSV as dictionaries
        reader = DictReader(file)
        # Create default styles list
        default_styles = [
            {"name": "None", "prompt": "{prompt}", "negative_prompt": ""},
        ]
        # Read in styles from CSV into styles list
        styles = list(reader)
        # Insert default styles at start of styles list
        for item in default_styles:
            styles.insert(0, item)

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in styles}
default_style = styles["None"]
style_keys = list(styles.keys())


SD_XL_BASE_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1152, 896),
    "3:2": (1216, 832),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:4": (896, 1152),
    "2:3": (832, 1216),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


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
