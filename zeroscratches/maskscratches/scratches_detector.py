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
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import ImageFile, Image

from zeroscratches.maskscratches.detection_models import networks
from zeroscratches.util import data_transforms, scale_tensor, tensor_to_ndarray

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ScratchesDetector:

    def __init__(self, snapshot_folder):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_mask = networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )

        model_path = os.path.join(snapshot_folder, "detection/FT_Epoch_latest.pt")
        checkpoint = torch.load(model_path, map_location=device)
        self.model_mask.load_state_dict(checkpoint["model_state"])
        self.model_mask.cpu()
        self.model_mask.eval()

    def process(self, image: Image) -> np.array:
        logging.info("Start detecting scratches")
        transformed_image = data_transforms(image, size="full_size")
        image = transformed_image.convert("L")
        image = tv.transforms.ToTensor()(image)
        image = tv.transforms.Normalize([0.5], [0.5])(image)
        image = torch.unsqueeze(image, 0)
        _, _, ow, oh = image.shape
        scratch_image_scale = scale_tensor(image)

        scratch_image_scale = scratch_image_scale.cpu()
        with torch.no_grad():
            prediction = torch.sigmoid(self.model_mask(scratch_image_scale))

        prediction = prediction.data.cpu()
        prediction = F.interpolate(prediction, [ow, oh], mode="nearest")

        tensor_mask = (prediction >= 0.4).float()
        scratches_mask_image = tensor_to_ndarray(tensor_mask)
        transformed_image = np.array(transformed_image)
        return transformed_image, scratches_mask_image
