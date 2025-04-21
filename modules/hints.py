import gradio as gr
import random
import glob


def get_hint():
    hintfiles = glob.glob("hints/*.txt")
    hints = []
    for hintfile in hintfiles:
        lines = open(hintfile, encoding='utf8').read().splitlines()
        hints += lines

    hint = f"**LPT:** *{random.choice(hints)}*"
    return hint
