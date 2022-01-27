from typing import Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from settings import model_settings


def edl_loss(type: str = 'mse') -> Callable:
    """
    Returns the selected EDL Loss
    """
    if type=='mse':
        return EDLMSELoss()
    raise NotImplementedError


class EDLMSELoss(nn.Module):
    """
    Evidential Deep Learning loss based on Mean Squared Error
    as defined in the cited paper
    """

    def __init__(self):
        super().__init__()

    def forward(self, alpha: Tensor, target: Tensor) -> Tensor:
        dirichlet_strength = alpha.sum(dim=1, keepdim=True)
        one_hot_target = F.one_hot(target, num_classes=model_settings.NUM_CLASSES)
        prediction_error = torch.sum(
                (one_hot_target - alpha / dirichlet_strength)**2, 
                dim=1, 
                keepdim=True
            )
        prediction_variance = torch.sum(
                alpha * (dirichlet_strength - alpha)
                /(dirichlet_strength * dirichlet_strength * (dirichlet_strength + 1)), 
                dim=1, 
                keepdim=True
            )
        return (prediction_error + prediction_variance).mean()

