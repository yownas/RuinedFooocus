import gradio as gr
import random
import glob
from modules.settings import default_settings

def get_hint():
    rnd = random.randint(0, 100)
    if default_settings["hint_chance"] > rnd:
        hintfiles = glob.glob("hints/*.txt")
        hints = []
        for hintfile in hintfiles:
            lines = open(hintfile).read().splitlines()
            hints += lines

        print(f"DEBUG: {hints}")
        hint = random.choice(hints)
        gr.Info(hint)

