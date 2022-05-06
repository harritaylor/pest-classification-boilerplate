import os
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torch import nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image


#  
class PestClassificationDataset(data.Dataset):
    '''
    https://pytorch.org/docs/stable/data.html
    All subsequent operations which use a DataLoader require the data in Pytorch's Dataset class.
    The base requirements are to implement the 3 following definitions:
    - __init__() to define labels, and hooks for data and label transforms
    - __len__() to return the length of the instanstiated dataset
    - __getitem__(idx) to return a tuple of [Data, Target] at a given idx
    '''
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.labels = [
            'mictis_longicornis',
            'apoderus_javanicus',
            'valanga_nigricornis',
            'dappula_tertia',
            'normal',
            'neomelicharia_sparsa',
            'dialeuropora_decempuncta',
            'icerya_seychellarum',
            'procontarinia_matteiana',
            'procontarinia_rubus',
            'orthaga_euadrusalis',
            'cisaberoptus_kenyae',
            'erosomyia_sp',
            'aulacaspis_tubercularis',
            'ceroplastes_rubens',
            'ischnaspis_longirostris'
        ]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        label = self.labels.index(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class PestClassificationDataModule(pl.LightningDataModule):
    '''
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    LightningDataModules are a way of organising all data related code into a 
    reusable module. This is not required but makes things a whole lot easier to keep
    track of! The three required things are:
    - __init__() - define the transforms, hooks for batch_size, and where the data is stored on disk
    - setup() - to instansiate the respective datasets, self.train, self.val (optional) and self.test.
                This is using an instance of the Dataset class we defined earlier.
    - {train, val, test}_dataloader: defining instances of PyTorch dataloaders
    '''
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # feel free to change these augmentations, they are just copied and pasted from the torchvision
        # docs. The important bit is the Normalize() which should norm the RGB values to match ImageNet
        # (you might want to double check I put the right values here...)
        # The other augmentations ensure that whatever size input image you have, only a patch will be
        # used which will still match the input to the model. Data augmentation is important in training!
        # Don't skip it!
        self.augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        # Again, the normalisation step is important. But the key thing that will break the code is 
        # CenterCrop(size=224). ResNets (the model we are using later) pre-trained on imagenet use an input size
        # of 224x224.
        self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize(size=256),
                                                transforms.CenterCrop(size=224),
                                                transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]
                                            )])

    def setup(self, stage=None):
        train_dataset = PestClassificationDataset(annotations_file=f'{self.data_dir}/training_set.csv', img_dir=f'{self.data_dir}/training_dataset')

        self.train, self.val = random_split(train_dataset, [280, 30]) # 280 training images, 30 val images. You can remove self.val if you want!
        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test = PestClassificationDataset(annotations_file=f'{self.data_dir}/Testing_set.csv', img_dir=f'{self.data_dir}/test')
        self.test.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

class TransferLearningModel(pl.LightningModule):
    '''
    Again, these are just random values I plucked from the internet - please experiment with them!
    You can use something like https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
    to automagically find the optimal values for learning rate etc. 
    '''
    def __init__(self, learning_rate=2e-4):
        super().__init__()

        self.learning_rate = learning_rate

        # using a pre-trained resnet to extract features.
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 16
        # using a single "fully-connected" layer to make predictions based off the 
        # penultimate resnet layers representations
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        # make sure that the gradient doesn't effect the existing pre-trained resnet.
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        # out of scope of `no_grad` == the layer will be trained!
        x = self.classifier(representations)
        x = F.relu(x)
        return x

    def configure_optimizers(self):
        # Again, other optimizers inside the `optim` package can be used, but Adam is
        # perfectly cromulent
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # This gives you an opportunity to do funky research stuff with the training step.
        # cross entropy loss is for single-predictions.
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # note that we don't return anything here, we just log it to the command line to make sure
        # that loss is decreasing
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # if test_loss and val_loss are not decreasing at the same rate then something is going really
        # wrong!!
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", val_loss)
        
        # TODO: you can add accuracy measures in both the validation_step and test_step here


if __name__ == "__main__":
    model = TransferLearningModel()
    dataset = PestClassificationDataModule(batch_size=28, data_dir='./pest_classification')
    
    '''
    TODO: you can add *many* options to the trainer: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    at the moment the code doesn't care if you have a GPU. From the docs...
    
    from argparse import ArgumentParser

    def main(hparams):
        model = LightningModule()
        trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
        trainer.fit(model)

    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--accelerator", default=None)
        parser.add_argument("--devices", default=None)
        args = parser.parse_args()

        main(args)
    '''
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=dataset)
