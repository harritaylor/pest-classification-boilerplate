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


class PestClassificationDataset(data.Dataset):
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
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size


        self.augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize(size=256),
                                                transforms.CenterCrop(size=224),
                                                transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]
                                            )])

    def setup(self, stage=None):
        train_dataset = PestClassificationDataset(annotations_file=f'{self.data_dir}/training_set.csv', img_dir=f'{self.data_dir}/training_dataset')

        self.train, self.val = random_split(train_dataset, [280, 30])
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
    def __init__(self, learning_rate=2e-4):
        super().__init__()

        self.learning_rate = learning_rate

        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 16
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = F.relu(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", val_loss)


if __name__ == "__main__":
    model = TransferLearningModel()
    dataset = PestClassificationDataModule(batch_size=28, data_dir='./pest_classification')
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=dataset)
