gradio_root = None

state = {
    "preview_grid": None,
    "ctrls_name": [],
    "ctrls_obj": []
}


def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]
