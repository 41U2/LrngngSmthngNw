import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import FER2013
from torchvision import transforms
from torch.utils.data import DataLoader

import lit_models.dense_classifier
from lit_models.vgg_classifier import VGGClassificationModel
from lit_models.resnet_classifier import ResNetClassificationModel
from lit_models.fmm_generator import FMMGenerationModel


n_epochs = 60
transforms_set = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,),
                                                       (0.3081,))])
train_data = FER2013(
    root="data",
    transform=transforms_set)

train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True
)

masks_dir = "C:/Programms/LrngngSmthngNw/nn/FER2013/data/masks/"
masks_path = os.listdir(masks_dir)
for i_mask in range(len(masks_path)):
    masks_path[i_mask] = masks_dir + masks_path[i_mask]


logger = TensorBoardLogger('tb_logs', name='fer2013_masks')

classification_model = FMMGenerationModel()
classification_model.import_masks(masks_path)

trainer = pl.Trainer(max_epochs=n_epochs, logger=logger)
trainer.fit(classification_model, train_loader)