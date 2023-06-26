#############################################################################
#
#   Source from:
#   https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
#   Forked from:
#
#   Reimplemented by: Leonel Hernández
#
##############################################################################
import logging
import os.path

import PIL.Image
import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torchvision.transforms import transforms

from zeroscratches.erasescratches.models import Pix2PixHDModel_Mapping
from zeroscratches.erasescratches.options import Options
from zeroscratches.maskscratches import ScratchesDetector
from zeroscratches.util import irregular_hole_synthesize, tensor_to_ndarray

REPO_ID = "leonelhs/zeroscratches"


class EraseScratches:

    def __init__(self):

        snapshot_folder = snapshot_download(repo_id=REPO_ID)
        model_path = os.path.join(snapshot_folder, "restoration")
        self.detector = ScratchesDetector(snapshot_folder)
        gpu_ids = []
        if torch.cuda.is_available():
            gpu_ids = [d for d in range(torch.cuda.device_count())]
        self.options = Options(model_path, gpu_ids)
        self.model_scratches = Pix2PixHDModel_Mapping()
        self.model_scratches.initialize(self.options)
        self.model_scratches.eval()

    def erase(self, image) -> np.array:
        transformed, mask = self.detector.process(image)
        logging.info("Start erasing scratches")

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        mask_transform = transforms.ToTensor()

        if self.options.mask_dilation != 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = np.array(mask)
            mask = cv2.dilate(mask, kernel, iterations=self.options.mask_dilation)
            mask = PIL.Image.fromarray(mask.astype('uint8'))

        transformed = irregular_hole_synthesize(transformed, mask)
        mask = mask_transform(mask)
        mask = mask[:1, :, :]  # Convert to single channel
        mask = mask.unsqueeze(0)
        transformed = img_transform(transformed)
        transformed = transformed.unsqueeze(0)

        try:
            with torch.no_grad():
                generated = self.model_scratches.inference(transformed, mask)
        except Exception as ex:
            raise TypeError("Skip photo due to an error:\n%s" % str(ex))

        tensor_restored = (generated.data.cpu() + 1.0) / 2.0
        return tensor_to_ndarray(tensor_restored)
