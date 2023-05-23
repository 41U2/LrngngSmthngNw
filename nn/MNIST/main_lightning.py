import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import lit_models.dense_classifier
import lit_models.vgg_classifier

n_epochs = 10
transforms_set = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,),
                                                       (0.3081,))])
train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms_set)
test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms_set)

train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=4,
    shuffle=False
)

logger = TensorBoardLogger('tb_logs', name='tensor_board_run_name')

number_classification_model = lit_models.vgg_classifier.VGGNumberClassificationModel()
trainer = pl.Trainer(max_epochs=n_epochs, logger=logger)
trainer.fit(number_classification_model, train_loader)
trainer.test(number_classification_model, test_loader)

