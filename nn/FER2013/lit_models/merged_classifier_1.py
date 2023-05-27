from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
import cv2
import numpy as np

from lit_models.fmm_generator import FMMGenerator, class_numbers_to_masks
from lit_models.resnet_classifier import ResNetClassifier, class_numbers_to_arrays


def load_image(path_to_image: str, device: str) -> torch.Tensor:
    image = cv2.imread(path_to_image)
    image = cv2.resize(image, [48, 48])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[::-1, :] / 255
    return torch.tensor(image, dtype=torch.float32, device=device)


def from_torch_to_cv(image: torch.Tensor) -> np.array:
    result_image = image.detach().cpu().numpy()[::-1, :, :]
    result_image = np.clip(np.rint(result_image * 255), 0, 255).astype(np.uint8)
    return result_image


class MergedClassificationModel(pl.LightningModule):

    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.fmmg = FMMGenerator()
        self.classification_net = ResNetClassifier(n_classes)
        self.loss_func = nn.MSELoss()
        self.masks = []
        self.n_classes = n_classes

    def mask(self, x:torch.Tensor) -> torch.Tensor:
        return self.fmmg.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification_net.forward(x + self.mask(x))

    def import_masks(self, pathes: List[str]):
        masks = []
        for path in pathes:
            image = load_image(path, "cuda")
            masks = masks + [image]
        self.masks = masks

    def training_step(self, batch, batch_idx):
        x, y = batch
        masks = self.fmmg.forward(x)
        y_masks = class_numbers_to_masks(y, self.masks)
        y_array = class_numbers_to_arrays(y, self.n_classes)
        pred = self.classification_net.forward(masks + x)
        loss = self.loss_func(pred, y_array) + self.loss_func(masks, y_masks) * 7. / (48. * 48.)
        self.log('train_loss', loss)

        # images_dir = "C:/Programms/LrngngSmthngNw/nn/FER2013/data/fer2013/emotions/"
        # for i_image in range(len(x)):
        #     image_to_write = torch.unsqueeze(torch.squeeze(x[i_image]), 2)
        #     cv2.imwrite(images_dir + str(y[i_image].cpu().numpy()) + ".jpg", from_torch_to_cv(image_to_write))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # y_1 = class_numbers_to_arrays(y, self.n_classes)
        # pred = self.net.forward(x)
        # loss = self.loss_func(pred, y_1)
        # pred = torch.argmax(pred, dim=1)
        # accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        #
        # self.log('test_loss', loss, prog_bar=True)
        # self.log('test_acc', torch.tensor(accuracy), prog_bar=True)
        # output = dict({
        #     'test_loss': loss,
        #     'test_acc': torch.tensor(accuracy),
        # })
        #
        # return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer