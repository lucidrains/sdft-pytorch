from __future__ import annotations
from typing import Callable

import torch
from torch import nn, Tensor
from torch.nn import Module

from einops import rearrange

from ema_pytorch import EMA

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class SDFT(Module):
    def __init__(
        self,
        model: Module,
        ema_alpha = 0.01
    ):
        super().__init__()

        self.student = model
        self.teacher = EMA(model, include_online_model = False)

    def forward(
        self
    ):
        raise NotImplementedError
