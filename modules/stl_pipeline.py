import modules.async_worker as worker
from modules.settings import default_settings

from modules.util import generate_temp_filename
import shared

import torch
import numpy as np
import os
import rembg
from PIL import Image
from pathlib import Path

from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground

import trimesh
import pyrender

class pipeline:
    pipeline_type = ["stl"]
    model_hash = ""

    pipeline = None
    model = None
    device = "cuda:0"

    # Optional function
    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name):
        if self.pipeline is not None:
            return

        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            self.device = "cpu"

        print('Loading SF3D model ...')

        worker.outputs.append(["preview", (-1, f"Loading SF3D model ...", None)])
        os.environ["HF_HOME"] = "models/diffusers_cache"
        if "img2stl_repo" in default_settings:
            repo = default_settings["img2stl_repo"]
        else:
            repo = "stabilityai/stable-fast-3d"
        self.pipeline = SF3D.from_pretrained(
            repo,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self.pipeline.to(self.device)
        self.pipeline.eval()

        print('Loading Finished!')
        return

    def load_keywords(self, lora):
        return " "

    def load_loras(self, loras):
        return

    def refresh_controlnet(self, name=None):
        return

    def clean_prompt_cond_caches(self):
        return

    def process(
        self,
        positive_prompt,
        negative_prompt,
        input_image,
        controlnet,
        main_view,
        steps,
        width,
        height,
        image_seed,
        start_step,
        denoise,
        cfg,
        sampler_name,
        scheduler,
        clip_skip,
        callback,
        gen_data=None,
    ):
        # GIFs parameters
        frame_count = 40
        duration_frame = 0.1

        # Visualization parameters
        init_angle = 0
        elevation = 20
        rotation_axises = [1.0, 0.0, 0.0]
        rotation_angle = -90
        x_offset = 0
        y_offset = 0
        z_offset = 0

        if gen_data["prompt"].startswith("token:"):
            from huggingface_hub import login
            login(token=gen_data["prompt"].split("token:")[1].strip())

        worker.outputs.append(["preview", (-1, f"Removing background ...", None)])

        rembg_session = rembg.new_session()
        input_image = remove_background(
            input_image.convert("RGBA"), rembg_session
        )
        input_image = resize_foreground(input_image, 0.85)

        if callback is not None:
            callback(0, 0, 0, 0, input_image)

        stl_filename = generate_temp_filename(
            folder=shared.path_manager.model_paths["temp_outputs_path"],
            extension="stl",
        )
        images = generate_temp_filename(
            folder=shared.path_manager.model_paths["temp_outputs_path"],
            extension="gif",
        )
        dir_path = Path(stl_filename).parent
        dir_path.mkdir(parents=True, exist_ok=True)

        worker.outputs.append(["preview", (-1, f"Running SF3D ...", None)])

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            texture_resolution = 1024
            remesh_option = "none" # choices=["none", "triangle", "quad"],
            sf3d_mesh, glob_dict = self.pipeline.run_image(
                [input_image],
                bake_resolution=texture_resolution,
                remesh=remesh_option,
            )

        worker.outputs.append(["preview", (-1, f"Export STL ...", None)])
        sf3d_mesh.export(stl_filename)
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        print("Creating GIF")
        scene = pyrender.Scene(ambient_light=[0.02, 0.03, 0.03],bg_color=[0.0, 0.0, 0.0])
        mesh = trimesh.load(stl_filename)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        light1 = pyrender.PointLight(color=[0.0, 1.0, 1.0], intensity=10.0)
        light2 = pyrender.PointLight(color=[1.0, 0.0, 0.0], intensity=10.0)

        # Set the 4x4 transformation matrices for the camera and the box surface. Start with an eye matrix and change the translation
        cam_matrix = np.eye(4)
        cam_matrix[:3, 3] = np.array([0,0,2])

        light1_matrix = np.array([
            [1., 0., 0., 2.],
            [0., 1., 0., 2.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.]
        ])

        light2_matrix = np.array([
            [1., 0., 0., 2.],
            [0., 1., 0., -3.],
            [0., 0., 1., 1.5],
            [0., 0., 0., 1.]
        ])

        nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        nc = pyrender.Node(camera=cam, matrix=cam_matrix)
        nl1 = pyrender.Node(light=light1, matrix=light1_matrix)
        nl2 = pyrender.Node(light=light2, matrix=light2_matrix)

        scene.add_node(nm)
        scene.add_node(nc)
        scene.add_node(nl1)
        scene.add_node(nl2)

        viewer_options = {"rotate_axis": [0,1,0]}
        render_options = {"face_normals":False}
        # We explicitly set raymond lighting - aka connected to the camera and set the flag for running the viewer in a separate thread so we can animate the objects
#        v = pyrender.Viewer(scene, use_raymond_lighting = True, viewer_flags = viewer_options, render_flags = render_options, run_in_thread=True )
        r = pyrender.OffscreenRenderer(
            viewport_width=640,
            viewport_height=480,
            point_size=1.0,
        )

        worker.outputs.append(["preview", (-1, f"Create GIF ...", None)])

        yaxis = [0., 1., 0.]

        frames = []
        flags = pyrender.constants.RenderFlags.FACE_NORMALS | pyrender.constants.RenderFlags.SHADOWS_POINT
        for i in range(frame_count):    
            R = trimesh.transformations.rotation_matrix((np.pi*2.*(frame_count-i))/frame_count, yaxis)
            scene.set_pose(nm, R)
            color, _ = r.render(scene)
            frames.append(Image.fromarray(color))

        # Save the images as a GIF using imageio
        os.makedirs(os.path.dirname(images), exist_ok=True)
        frames[0].save(
            images,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_frame,
            loop=0,
        )

        return [images]
