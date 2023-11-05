import os
import sys
import cv2
import modules.path
import modules.async_worker as worker

from modules.settings import default_settings
from modules.util import suppress_stdout

import warnings
from PIL import Image
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
            model_path = os.path.join(modules.path.insightface_path, model_name)
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
    ):
        worker.outputs.append(["preview", (-1, f"Generating ...", None)])

        warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
        warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

        input_image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
        output_image = cv2.imread(progress_window)

        input_faces = sorted(self.analyser_model.get(input_image), key=lambda x: x.bbox[0])
        output_faces = sorted(self.analyser_model.get(output_image), key=lambda x: x.bbox[0])

        frame = output_image
        idx = 0
        for output_face in output_faces:
            frame = self.swapper_model.get(frame, output_face, input_faces[idx%len(input_faces)], paste_back=True)
            idx+=1

        images = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        # Return finished image to preview
        if callback is not None:
            callback(steps, 0, 0, steps, images)

        return [images]
