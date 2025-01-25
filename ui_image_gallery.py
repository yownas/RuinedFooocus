import gradio as gr
from shared import path_manager
from modules.imagebrowser import ImageBrowser

browser = ImageBrowser()


def create_image_gallery():
    with gr.Blocks(theme=gr.themes.Soft()) as app_image_browser:
        with gr.Row():
            # Left side for gallery
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Images",
                    show_label=False,
                    columns=[4],
                    height="600px",
                    object_fit="contain",
                    value=browser.load_images(1)[0],
                )
                ib_page = gr.Slider(
                    label="Page",
                    value=1,
                    step=1,
                    minimum=1,
                    maximum=browser.num_images_pages()[1],
                )
                ib_range = gr.Markdown()

            # Right side for metadata and search
            with gr.Column(scale=1):
                with gr.Row():
                    update_btn = gr.Button("Update DB", scale=5)
                    gr.HTML(value="""<a href="gradio_api/file/html/slideshow.html" style="color: gray; text-decoration: none" target="_blank">🛝</a>""")
                metadata_output = gr.Textbox(
                    label="Image Metadata", interactive=False, lines=15
                )
                search_input = gr.Textbox(
                    label="Search Metadata", placeholder="Enter search term"
                )
                search_btn = gr.Button("Search")
                status_output = gr.Markdown()

        # Event handlers
        update_btn.click(
            browser.update_images, inputs=[], outputs=[gallery, ib_page, status_output]
        )
        ib_page.change(
            browser.load_images, inputs=[ib_page], outputs=[gallery, ib_range]
        )
        gallery.select(browser.get_image_metadata, None, metadata_output)
        search_btn.click(
            browser.search_metadata,
            inputs=[search_input],
            outputs=[gallery, ib_page, status_output],
        )
        search_input.submit(
            browser.search_metadata,
            inputs=[search_input],
            outputs=[gallery, ib_page, status_output],
        )


    return app_image_browser


# Optional: If you want to launch just this gallery
def launch_image_gallery():
    app_image_browser = create_image_gallery()
    app_image_browser.launch()
