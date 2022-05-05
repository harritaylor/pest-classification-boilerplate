import os
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image


class PestClassificationDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
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

        self.num_classes = ...

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.transform = transforms.Compose([
                                                transforms.Resize(size=256),
                                                transforms.CenterCrop(size=224),
                                                transforms.ToTensor(),

            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                            ])

    def setup(self, stage=None):
        train_dataset = PestClassificationDataset(annotations_file=f'{self.data_dir}/training_set.csv', img_dir=f'{self.data_dir}/training_dataset')

        self.train, self.val = random_split(train_dataset, [280,30])
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

def main():
    print("Hello, World!")
    ds = PestClassificationDataModule(batch_size=12, data_dir='./pest_classification')
    ds.setup()
    breakpoint()


if __name__ == "__main__":
    main()
