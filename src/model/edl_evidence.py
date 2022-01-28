from typing import Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.sparse import softmax


def logits_to_evidence(type: str = 'relu') -> Callable:
    """
    Returns the desired method of logit conversion
    to non-negative 
    """
    if type=='relu':
        return relu_evidence
    if type=='exp':
        return exp_evidence
    if type=='softplus':
        return softplus_evidence
    raise NotImplementedError


def relu_evidence(logits: Tensor) -> Tensor:
    """
    Logit conversion to non-negative values using ReLU
    """
    return F.relu(logits)


def exp_evidence(logits: Tensor) -> Tensor:
    """
    Logit conversion to non-negative values using exponentials
    """
    return torch.exp(torch.clip(logits, -10, 10))


def softplus_evidence(logits: Tensor) -> Tensor:
    """
    Logit conversion to non-negative values using Softplus
    """
    return nn.Softplus()(logits)
