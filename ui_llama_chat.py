import gradio as gr
from shared import path_manager
import modules.async_worker as worker

def create_chat():
    with gr.Blocks() as app_llama_chat:
        with gr.Group():
            llama_chat = gr.Chatbot(
                label="Ruined Chatbot",
                type="messages",
            )
            llama_msg = gr.Textbox(
                show_label=False,
            )

        def llama_respond(message, chat_history):
            chat_history.append({"role": "user", "content": message})

            gen_data = {
                "task_type": "llama",
                "system": "",
                "history": chat_history,
            }

            # Add work
            task_id = worker.add_task(gen_data.copy())

            # Wait for result
            finished = False
            while not finished:
                flag, product = worker.task_result(task_id)
                if flag == "results":
                    finished = True

            chat_history.append({"role": "assistant", "content": product})
            return "", chat_history


        llama_msg.submit(llama_respond, [llama_msg, llama_chat], [llama_msg, llama_chat])

    return app_llama_chat

