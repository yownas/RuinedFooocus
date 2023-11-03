gradio_root = None

state = {"preview_image": None, "ctrls_name": [], "ctrls_obj": [], "pipeline": None}


def add_ctrl(name, obj):
    state["ctrls_name"] += [name]
    state["ctrls_obj"] += [obj]
