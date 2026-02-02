from __future__ import annotations
from typing import Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from jinja2 import Environment, meta

from einops import rearrange

from ema_pytorch import EMA

# default query / demonstration template for in-context learned distillation targets from teacher for student

DEFAULT_TEACHER_PROMPT_TEMPLATE = """
[Task Instructions] You are a helpful assistant. Please answer the question based on the provided logic.

[Expert Demonstration] Question: {{ question }} Expert Reasoning and Answer: {{ answer }}

[Current Task] Question: {{ question }} Answer:
"""

def get_variables_from_template(template):

    env = Environment()

    parsed_template = env.parse(template)

    return set(meta.find_undeclared_variables(parsed_template))

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
        teacher_update_rate = 0.01,
        teacher_prompt_template = DEFAULT_TEACHER_PROMPT_TEMPLATE,
    ):
        super().__init__()

        self.student = model

        self.teacher = EMA(
            model,
            beta = 1. - teacher_update_rate,
            include_online_model = False
        )

        # store teacher template

        assert get_variables_from_template(teacher_prompt_template) == {'question', 'answer'}, 'your template must contain only variables `question` and `answer`, embedded like so - {{ question }} ... {{ answer }}'
        self.teacher_prompt_template = teacher_prompt_template

    def forward(
        self,
        question,
        answer
    ):
        raise NotImplementedError
