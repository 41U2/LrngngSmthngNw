import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
from torchvision.datasets import FER2013
from torchvision import transforms
from torch.utils.data import DataLoader

import lit_models.dense_classifier
import lit_models.vgg_classifier

transforms_set = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
     ])
test_data = torchvision.datasets.FER2013(
    root="data",
    split="test",
    transform=transforms_set
)

test_loader = DataLoader(
    test_data,
    batch_size=4,
    shuffle=False
)

number_classification_model = lit_models.vgg_classifier.VGGClassificationModel(7).load_from_checkpoint(
    "tb_logs/fer2013/version_1/checkpoints/epoch=34-step=251230.ckpt")
trainer = pl.Trainer(max_epochs=10)
trainer.test(number_classification_model, test_loader)
