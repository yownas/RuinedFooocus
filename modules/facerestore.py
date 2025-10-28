import os
import sys
import cv2
import re
from shared import path_manager
import modules.async_worker as worker
from tqdm import tqdm

from modules.util import generate_temp_filename

from PIL import Image
import imageio.v3 as iio
import numpy as np
import torch
import insightface

from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader, module_from_spec

class ImportRedirector(MetaPathFinder):
    def __init__(self, redirect_map):
        self.redirect_map = redirect_map

    def find_spec(self, fullname, path, target=None):
        if fullname in self.redirect_map:
            return spec_from_loader(fullname, ImportLoader(self.redirect_map[fullname]))
        return None

class ImportLoader(Loader):
    def __init__(self, redirect):
        self.redirect = redirect

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import importlib
        redirected = importlib.import_module(self.redirect)
        module.__dict__.update(redirected.__dict__)

# Set up the redirection
redirect_map = {
    'torchvision.transforms.functional_tensor': 'torchvision.transforms.functional'
}

sys.meta_path.insert(0, ImportRedirector(redirect_map))

import gfpgan
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

# Models in models/faceswap/
# https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

class facerestore:
    gfpgan_model = None

    def load_gfpgan_model(self):
        if self.gfpgan_model is None:
            channel_multiplier = 2

            model_name = "GFPGANv1.4.pth"
            model_path = os.path.join(path_manager.model_paths["faceswap_path"], model_name)

            # https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
            self.gfpgan_model = gfpgan.GFPGANer
            self.gfpgan_model.bg_upsampler = None
            # initialize model
            self.gfpgan_model.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            upscale = 2
            self.gfpgan_model.face_helper = FaceRestoreHelper(
                upscale,
                det_model="retinaface_resnet50",
                model_rootpath=path_manager.model_paths["faceswap_path"],
            )

            self.gfpgan_model.gfpgan = gfpgan.GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )

            loadnet = torch.load(model_path)
            if "params_ema" in loadnet:
                keyname = "params_ema"
            else:
                keyname = "params"
            self.gfpgan_model.gfpgan.load_state_dict(loadnet[keyname], strict=True)
            self.gfpgan_model.gfpgan.eval()
            self.gfpgan_model.gfpgan = self.gfpgan_model.gfpgan.to(
                self.gfpgan_model.device
            )

    def restore_faces(self, image):
        self.load_gfpgan_model()

        image_bgr = image[:, :, ::-1]
        _cropped_faces, _restored_faces, gfpgan_output_bgr = self.gfpgan_model.enhance(
            self.gfpgan_model,
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5,
        )
        image = gfpgan_output_bgr[:, :, ::-1]

        return image

    def process(self, input_image):
        input_image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
        result_image = self.restore_faces(input_image)
        image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

        return image
