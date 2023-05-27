from typing import List

import pytorch_lightning as pl
import torch
import cv2
from torch import nn


def class_numbers_to_masks(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    result = torch.zeros([len(x), len(masks[0]), len(masks[0][0])], device="cuda", dtype=torch.float32)
    for i_example in range(len(x)):
        result[i_example] = masks[x[i_example]]
    return result


def load_image(path_to_image: str, device: str) -> torch.Tensor:
    image = cv2.imread(path_to_image)
    image = cv2.resize(image, [48, 48])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[::-1, :] / 255
    return torch.tensor(image, dtype=torch.float32, device=device)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, (3, 3), padding=1)

    def forward(self, x: torch.Tensor):
        conved = self.conv2(self.conv1(x))
        return x + conved


class FMMGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(

            nn.Conv2d(1, 1, (3, 3)),
            #nn.MaxPool2d((2, 2)),

            nn.Conv2d(1, 1, (3, 3)),
            #nn.MaxPool2d((2, 2)),

            nn.Conv2d(1, 1, (3, 3)),
            #nn.MaxPool2d((2, 2)),

            ResidualBlock(1),
            ResidualBlock(1),
            ResidualBlock(1),
            ResidualBlock(1),

            nn.ConvTranspose2d(1, 1, (3, 3)),
            nn.ConvTranspose2d(1, 1, (3, 3)),
            nn.ConvTranspose2d(1, 1, (3, 3))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FMMGenerationModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = FMMGenerator()
        self.loss_func = nn.MSELoss()
        self.masks = []

    def import_masks(self, pathes: List[str]):
        masks = []
        for path in pathes:
            image = load_image(path, "cuda")
            masks = masks + [image]
        self.masks = masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = class_numbers_to_masks(y, self.masks)
        pred = self.net.forward(x)
        loss = self.loss_func(pred, y)
        self.log('train_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = class_numbers_to_masks(y, self.n_classes)
        pred = self.net.forward(x)
        loss = self.loss_func(pred, y)
        # pred = torch.argmax(pred, dim=1)
        # accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)

        self.log('test_loss', loss, prog_bar=True)
        # self.log('test_acc', torch.tensor(accuracy), prog_bar=True)
        output = dict({
            'test_loss': loss,
            # 'test_acc': torch.tensor(accuracy),
        })

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer