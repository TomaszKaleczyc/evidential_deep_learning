from typing import List, Tuple
from numpy.lib.recfunctions import _keep_fields

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchmetrics
import pytorch_lightning as pl


from model.base_model import BaseModel
from model import edl_losses, edl_utils 
from settings import model_settings


class LeNetEDL(pl.LightningModule):
    """
    Convolutional net using Evidential Deep Learning
    output to estimate the Dirichlet 
    """

    def __init__(self, logits_to_evidence_function: str = 'relu', loss_function: str = 'mse'):
        super().__init__()
        self.base_model = BaseModel(dropout_rate=model_settings.DROPOUT_RATE)
        self.accuracy = torchmetrics.Accuracy()
        self.logits_to_evidence_function = edl_utils.logits_to_evidence(logits_to_evidence_function)
        self.loss_function = edl_losses.edl_loss(loss_function)

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
        alpha, predicted_probabilities, _ = self._calculate_edl_factors(input_images)
        loss = self.loss_function(alpha, expected_classes)
        self.accuracy(predicted_probabilities, expected_classes)
        self.log(f'{dataset_name}/loss', loss)
        self.log(f'{dataset_name}/accuracy', self.accuracy)
        return loss

    def _calculate_edl_factors(self, image_batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Performs necessary EDL calculations as described in the paper
        """
        predicted_logits = self(image_batch)   
        evidence = self.logits_to_evidence_function(predicted_logits)
        alpha = evidence + 1
        dirichlet_strength = alpha.sum(dim=1, keepdim=True)
        uncertainty = model_settings.NUM_CLASSES / dirichlet_strength
        predicted_probabilities = alpha / dirichlet_strength
        return alpha, predicted_probabilities, uncertainty    

    def configure_optimizers(self):
        """
        Configuring the net optimization methods
        """
        return torch.optim.Adam(self.parameters())

    def predict(self, image_batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Zwraca prawdopodobieństwo na podstawie batcha zdjęć
        """
        _, predicted_probabilities, uncertainty = self._calculate_edl_factors(image_batch)
        predicted_classes = predicted_probabilities.argmax(dim=1)
        return predicted_classes, predicted_probabilities, uncertainty
