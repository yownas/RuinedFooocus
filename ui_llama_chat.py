import gradio as gr
from shared import path_manager
import modules.async_worker as worker
from pathlib import Path
import json


def create_chat():
    def llama_get_assistants():
        names = []
        folder_path = Path("chatbots")
        for path in folder_path.rglob("*"):
            if path.is_dir():
                try:
                    with open(path / "info.json" , "r") as f:
                        info = json.load(f)
                    names.append((info["name"], str(path)))
                except Exception as e:
                    print(f"ERROR: in {path}: {e}")
                    pass

        names.sort(key=lambda x: x[0].casefold())
        return names

    def llama_select_assistant(dropdown):
        folder = Path(dropdown)
        try:
            with open(folder / "info.json", "r") as f:
                info = json.load(f)
                if "avatar" not in info:
                    info["avatar"] = folder / "avatar.png"
        except Exception as e:
            print(f"ERROR: {dropdown}: {e}")
            info = {
                "name": "Error",
                "greeting": "Error!",
                "avatar": "html/error.png",
                "system": "Everything is broken.",
            }
            pass
        greeting = [{"role": "assistant", "content": info["greeting"]}]
        return {
            llama_chat: gr.update(value=greeting),
            llama_msg: gr.update(value=""),
            llama_avatar: gr.update(
                value=info["avatar"],
                label=info["name"],
            ),
            llama_system: gr.update(value=info["system"])
        }


    with gr.Blocks() as app_llama_chat:
        with gr.Row():
            with gr.Column(scale=3), gr.Group():
                # FIXME!!! start value should be read from some info.json
                llama_chat = gr.Chatbot(
                    label="Ruined Chatbot",
                    show_label=False,
                    height=600,
                    type="messages",
                    value=[
                        {"role": "assistant", "content": "What did you break now?"},
                    ],
                )
                llama_msg = gr.Textbox(
                    show_label=False,
                )
            with gr.Column(scale=2), gr.Group():
                llama_avatar = gr.Image(
                    value="html/error.png",
                    label="Ruined support-troll",
                    height=400,
                    width=400,
                    show_label=True,
                )
                llama_assistants = gr.Dropdown(
                    choices=llama_get_assistants(),
                    value="Ruined support-troll",
                    show_label=False,
                    interactive=True,
                )
                llama_system = gr.Textbox(
                    visible=False,
                    value="You are the Ruined Fooocus error troll. You should try to help the user but remember it is probably their fault it went wrong. You are angry and grouchy (but dont type in all caps) and speak in the style of a dwarf.",
                )

        def llama_respond(message, system, chat_history):
            chat_history.append({"role": "user", "content": message})

            gen_data = {
                "task_type": "llama",
                "system": system,
                "history": chat_history,
            }

            # Add work
            task_id = worker.add_task(gen_data.copy())

            # Wait for result
            finished = False
            while not finished:
                flag, product = worker.task_result(task_id)
                if flag == "preview":
                    yield "", product
                elif flag == "results":
                    finished = True

            chat_history.append({"role": "assistant", "content": product})
            yield "", chat_history

        llama_msg.submit(
            llama_respond, [llama_msg, llama_system, llama_chat], [llama_msg, llama_chat]
        )
        llama_assistants.select(llama_select_assistant, [llama_assistants], [llama_chat, llama_msg, llama_avatar, llama_system])

    return app_llama_chat
