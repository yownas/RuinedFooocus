import os
import sys
import cv2
import modules.path
import modules.async_worker as worker
from tqdm import tqdm
import tempfile

from modules.settings import default_settings
from modules.util import suppress_stdout, generate_temp_filename

import warnings
from PIL import Image
import imageio.v3 as iio
import numpy as np
import insightface
import onnxruntime


class pipeline():
    pipeline_type = ["faceswapper"]

    analyser_model = None
    analyser_hash = ""
    swapper_model = None
    swapper_hash = ""

    def load_base_model(self, name):
        model_name = "inswapper_128.onnx"
        if not self.swapper_hash == model_name:
            print(f"Loading swapper model: {model_name}")
            model_path = os.path.join(modules.path.faceswap_path, model_name)
            try:
                with open(os.devnull, 'w') as sys.stdout:
                    self.swapper_model = insightface.model_zoo.get_model(
                        model_path,
                        download=False,
                        download_zip=False,
                    )
                    self.swapper_hash = model_name
                sys.stdout = sys.__stdout__
            except:
                print(f"Failed loading model! {model_path}")

        model_name = "buffalo_l"
        det_thresh = 0.5
        if not self.analyser_hash == model_name:
            print(f"Loading analyser model: {model_name}")
            try:
                with open(os.devnull, 'w') as sys.stdout:
                    self.analyser_model = insightface.app.FaceAnalysis(name='buffalo_l')
                    self.analyser_model.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
                    self.model_hash = model_name
                sys.stdout = sys.__stdout__
            except:
                print(f"Failed loading model! {model_name}")

    def load_keywords(self, lora):
        return ""

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
        progress_window,
        steps,
        width,
        height,
        image_seed,
        start_step,
        denoise,
        cfg,
        sampler_name,
        scheduler,
        callback,
        gen_data=None,
    ):
        worker.outputs.append(["preview", (-1, f"Generating ...", None)])

        warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
        warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

        input_image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
        input_faces = sorted(self.analyser_model.get(input_image), key=lambda x: x.bbox[0])

        prompt = gen_data["prompt"].strip()
        if prompt.startswith("https://") and prompt.endswith(".gif"):
            x = iio.immeta(prompt)
            duration = x['duration']
            loop = x['loop']
            gif = cv2.VideoCapture(prompt) 

            # Swap
            in_imgs = []
            out_imgs = []
            while(True):
                ret, frame = gif.read()
                if not ret:
                    break
                in_imgs.append(frame)

            with tqdm(total=len(in_imgs), desc="Groop", unit="frames") as progress:
                for frame in in_imgs:
                    ##out_faces = select_faces(frame, None, det_thresh)
                    out_faces = sorted(self.analyser_model.get(frame), key=lambda x: x.bbox[0])

                    idx = 0
                    for out_face in out_faces:
                        ##if sim_thresh == 0 or tgt is not None and cosDist(tgt, out_face) <= sim_thresh:
                        frame = self.swapper_model.get(frame, out_face, input_faces[idx%len(input_faces)], paste_back=True)
                        ##if restore:
                        ##    frame = face_restoration.restore_faces(np.asarray(frame))
                        idx+=1
                    out_imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    progress.update(1)

            images = generate_temp_filename(
                    folder=modules.path.temp_outputs_path, extension="gif"
                )
            out_imgs[0].save(images, save_all=True, append_images=out_imgs[1:], optimize=True, duration=duration, loop=loop)

        else:
            output_image = cv2.imread(progress_window)
            output_faces = sorted(self.analyser_model.get(output_image), key=lambda x: x.bbox[0])

            frame = output_image
            idx = 0
            for output_face in output_faces:
                frame = self.swapper_model.get(frame, output_face, input_faces[idx%len(input_faces)], paste_back=True)
                idx+=1
            images = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        return [images]
