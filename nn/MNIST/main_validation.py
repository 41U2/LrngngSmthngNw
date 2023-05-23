import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import lit_models.dense_classifier
import lit_models.vgg_classifier

transforms_set = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,),
                                                       (0.3081,))])
test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms_set)

test_loader = DataLoader(
    test_data,
    batch_size=4,
    shuffle=False
)

number_classification_model = lit_models.vgg_classifier.VGGNumberClassificationModel().load_from_checkpoint(
    "tb_logs/tensor_board_run_name/version_2/checkpoints/epoch=9-step=150000.ckpt")
trainer = pl.Trainer(max_epochs=n_epochs)
trainer.test(number_classification_model, test_loader)