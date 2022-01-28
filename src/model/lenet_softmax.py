from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchmetrics
import pytorch_lightning as pl


from model.base_model import BaseModel
from settings import model_settings


class LeNetSoftmax(pl.LightningModule):
    """
    Classic convolutional architecture
    with Softmax activation at the end,
    aimed to estimate class probabilities
    """

    def __init__(self):
        super().__init__()
        self.base_model = BaseModel(dropout_rate=model_settings.DROPOUT_RATE)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, img: Tensor):
        output = self.base_model(img)
        return output

    def training_step(self, batch: List[Tensor], batch_idx: int):
        return self._loss_step(batch, batch_idx, dataset_name='train')

    def validation_step(self, batch: List[Tensor], batch_idx: int):
        return self._loss_step(batch, batch_idx, dataset_name='validation')

    def _loss_step(self, batch: List[Tensor], batch_idx: int, dataset_name: str):
        """
        Standard training/validation step
        """
        input_images, expected_classes = batch
        predicted_logits = self(input_images)   
        predicted_probabilities = torch.softmax(predicted_logits, dim=1)
        loss = F.cross_entropy(predicted_logits, expected_classes)
        self.accuracy(predicted_probabilities, expected_classes)
        self.log(f'{dataset_name}/loss', loss)
        self.log(f'{dataset_name}/accuracy', self.accuracy)
        return loss

    def configure_optimizers(self):
        """
        Configuring the net optimization methods
        """
        parameter_groups = [
            {'params': self.base_model.feature_extractor.parameters(), 'weight_decay': model_settings.FEATURE_EXTRACTOR_WEIGHT_DECAY},
            {'params': self.base_model.head.parameters(), 'weight_decay': model_settings.HEAD_WEIGHT_DECAY}
        ]
        return torch.optim.Adam(
            parameter_groups, 
            lr=model_settings.LEARNING_RATE
            )

    def predict(self, image_batch: List[Tensor]):
        """
        Zwraca prawdopodobieństwo na podstawie batcha zdjęć
        """
        predicted_logits = self(image_batch)   
        predicted_probabilities = torch.softmax(predicted_logits, dim=1)
        return predicted_probabilities.argmax(dim=1), predicted_probabilities
