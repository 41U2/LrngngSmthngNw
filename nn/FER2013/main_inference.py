import os
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
from torchvision.datasets import FER2013
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np

import lit_models.dense_classifier
from lit_models.vgg_classifier import VGGClassificationModel
from lit_models.resnet_classifier import ResNetClassificationModel
from lit_models.merged_classifier_1 import MergedClassificationModel


def normalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)


def load_custom_image(path_to_image: str, device: str) -> torch.Tensor:
    image = cv2.imread(path_to_image)
    image = cv2.resize(image, [48, 48])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[::-1, :] / 255

    out = torch.unsqueeze(torch.tensor(image, dtype=torch.float32, device=device), 0)
    out = normalize(out, [0.1307], [0.3081])
    return out


def from_torch_to_cv(image: torch.Tensor) -> np.array:
    result_image = image.detach().cpu().numpy()[::-1, :, :]
    result_image = np.clip(np.rint(result_image * 255), 0, 255).astype(np.uint8)
    return result_image


model = MergedClassificationModel(7).load_from_checkpoint(
    "tb_logs/fer2013/version_18/checkpoints/epoch=259-step=1866280.ckpt")

#images_dir = "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/fear/"
images_dir = "C:/Programms/LrngngSmthngNw/nn/FER2013/data/OurFaces/"

# image_pathes = [
#     "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/angry/Training_3908.jpg",
#     "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/disgust/Training_8819879.jpg",
#     "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/fear/Training_4285241.jpg",
#     "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/happy/Training_831592.jpg",
#     "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/sad/Training_2153766.jpg",
#     "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/train/surprise/Training_2153766.jpg"
# ]
image_pathes = os.listdir(images_dir)


save_dir = "C:/Programms/LrngngSmthngNw/nn/FER2013/data/OurFacesResult/"

for i_image in range(len(image_pathes)):
    path = images_dir + image_pathes[i_image]

    image_cv = cv2.imread(path)
    image_cv = cv2.resize(image_cv, [48, 48])
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    initial_image = load_custom_image(path, "cuda")
    image = torch.unsqueeze(initial_image, 0)
    output = model.forward(image)
    output_class = torch.argmax(output)
    output_mask = torch.unsqueeze(torch.squeeze(model.mask(image)), 2)
    print("Path: ", image_pathes[i_image], ", class: ", output_class)
    cv2.imwrite(save_dir + image_pathes[i_image], from_torch_to_cv(output_mask))

kek = 0


