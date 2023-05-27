import pytorch_lightning as pl
import torch
from torch import nn


def class_numbers_to_arrays(x: torch.Tensor, n_classes: int) -> torch.Tensor:
    result = torch.zeros([len(x), n_classes], device="cuda", dtype=torch.float32)
    for i_example in range(len(x)):
        result[i_example][x[i_example]] = 1
    return result


class VGGClassifier(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()
        self.body = nn.Sequential(

            nn.Conv2d(1, 3, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(3, 9, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(9, 27, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(27, 81, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(81, 162, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(162, 324, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),
            nn.Linear(1296, 1296),
            nn.ReLU(),
            nn.Linear(1296, n_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class VGGClassificationModel(pl.LightningModule):

    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.net = VGGClassifier(n_classes)
        self.loss_func = nn.MSELoss()
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = class_numbers_to_arrays(y, self.n_classes)
        pred = self.net.forward(x)
        loss = self.loss_func(pred, y)
        self.log('train_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_1 = class_numbers_to_arrays(y, self.n_classes)
        pred = self.net.forward(x)
        loss = self.loss_func(pred, y_1)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', torch.tensor(accuracy), prog_bar=True)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer