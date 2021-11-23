from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule

from settings import consts

torch.manual_seed(consts.RANDOM_SEED)


class MNISTDataModule(LightningDataModule):
    """
    
    """

    def __init__(self):
        super().__init__()
        self._train_batch_size = consts.BATCH_SIZE_TRAIN
        self._val_batch_size = consts.BATCH_SIZE_VAL
        self._normalisation_terms = (consts.MNIST_MEAN, consts.MNIST_STD)
        self._data_path = consts.DATA_PATH

    @property
    def transforms(self):
        """
        Defines the dataset transforms
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*self._normalisation_terms)
        ])

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        mnist_train_dataset = datasets.MNIST(self._data_path, train=True, download=True, transform=self.transforms)
        return DataLoader(mnist_train_dataset, batch_size=self._train_batch_size, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        mnist_val_dataset = datasets.MNIST(self._data_path, train=False, download=True, transform=self.transforms)
        return DataLoader(mnist_val_dataset, batch_size=self._val_batch_size, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        raise NotImplementedError
