from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module

from einops import rearrange
from torch_einops_utils import pack_with_inverse

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class SDFT(Module):
    def __init__(
        self
    ):
        super().__init__()
        raise NotImplementedError
