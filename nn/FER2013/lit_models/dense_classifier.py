import pytorch_lightning as pl
import torch
from torch import nn


def class_numbers_to_arrays(x: torch.Tensor, n_classes: int) -> torch.Tensor:
    result = torch.zeros([len(x), n_classes], device="cuda", dtype=torch.float32)
    for i_example in range(len(x)):
        result[i_example][x[i_example]] = 1
    return result


class DenseClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DenseClassificationModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = DenseClassifier()
        self.loss_func = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y = class_numbers_to_arrays(y, 10)
        pred = self.net.forward(x)
        loss = self.loss_func(pred, y)
        self.log('train_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y = class_numbers_to_arrays(y, 10)
        pred = self.net.forward(x)
        loss = self.loss_func(pred, y)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)

        # PyTorch
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', torch.tensor(accuracy), prog_bar=True)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        return optimizer
