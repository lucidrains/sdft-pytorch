from __future__ import annotations
from typing import Callable

from jinja2 import Template, Environment, meta

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, is_tensor, tensor, Tensor

from einops import rearrange

from torch_einops_utils import pad_sequence

from ema_pytorch import EMA

# default query / demonstration template for in-context learned distillation targets from teacher for student


DEFAULT_STUDENT_PROMPT_TEMPLATE = """
[Instruction]
You are a helpful assistant

[Query]
{{ question }}

[Response]
"""

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

def maybe_cast_tensor(t):
    return t if is_tensor(t) else tensor(t)

# classes

class SDFT(Module):
    def __init__(
        self,
        model: Module,
        tokenizer_encode: Callable[[list[str]], list[Tensor]],
        student_prompt_template = DEFAULT_STUDENT_PROMPT_TEMPLATE,
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

        # collection of prompts to list[Int['seq']]

        self.tokenizer_encode = tokenizer_encode

        # store templates

        assert get_variables_from_template(teacher_prompt_template) == {'question', 'answer'}, 'your template must contain only variables `question` and `answer`, embedded like so - {{ question }} ... {{ answer }}'
        self.teacher_prompt_template = Template(teacher_prompt_template)

        assert get_variables_from_template(student_prompt_template) == {'question'}
        self.student_prompt_template = Template(student_prompt_template)

    def forward(
        self,
        questions: list[str],
        answers: list[str]
    ):
        encode = self.tokenizer_encode
        assert len(questions) == len(answers)

        student_vars = [{'question': question} for question in questions]
        teacher_vars = [{'question': question, 'answer': answer} for question, answer in zip(questions, answers)]

        # ready the prompts for student and teacher

        student_prompts_str = [self.student_prompt_template.render(questions) for questions in student_vars]
        teacher_prompts_str = [self.teacher_prompt_template.render(question_answers) for question_answers in teacher_vars]

        student_prompt_ids = [maybe_cast_tensor(encode(prompt)) for prompt in student_prompts_str]
        teacher_prompt_ids = [maybe_cast_tensor(encode(prompt)) for prompt in teacher_prompts_str]

        student_prompt_ids, student_prompt_id_lens = pad_sequence(student_prompt_ids, return_lens = True, left = True, value = -1)
        teacher_prompt_ids, teacher_prompt_id_lens = pad_sequence(teacher_prompt_ids, return_lens = True, left = True, value = -1)

        return tensor(0.)
