import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision import transforms

class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 64,
        val_batch_size: int = 100,
        image_size: int = 224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size

        if os.name == 'nt':
            self.num_worker = 0
        else:
            self.num_worker = 1

    def prepare_data(self):
        # No action needed for preparing data
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                os.path.join(self.data_dir, "train"),
                transform=transforms.Compose(
                    [
                        transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                ),
            )
            self.val_dataset = ImageFolder(
                os.path.join(self.data_dir, "val"),
                transform=transforms.Compose(
                    [
                        transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                ),
            )
        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                os.path.join(self.data_dir, "test"),
                transform=transforms.Compose(
                    [
                        transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                ),
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_worker)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=self.num_worker)
