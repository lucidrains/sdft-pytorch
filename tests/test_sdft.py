import pytest
param = pytest.mark.parametrize

import torch
from torch.optim import Adam
from sdft_pytorch.sdft_pytorch import SDFT

@param('eos_id', (None, 1))
@param('num_init_student_response_tokens_mask', (0, 2))
def test_sdft(
    eos_id,
    num_init_student_response_tokens_mask
):
    from torch import tensor
    from x_transformers import TransformerWrapper, Decoder

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 512,
        attn_layers = Decoder(
            dim = 512,
            depth = 2
        )
    )

    def tokenizer_encode(prompts: list[str]):
        return [
            tensor([ord(c) for c in prompt])
            for prompt in prompts
        ]

    sdft_wrapper = SDFT(
        model,
        student_max_response_len = 128,
        eos_id = eos_id,
        tokenizer_encode = tokenizer_encode,
        num_init_student_response_tokens_mask = num_init_student_response_tokens_mask,
    )

    loss, response = sdft_wrapper(
        questions = ['12+48', '2*3'],
        answers = ['60', '6']
    )

    optim = Adam(sdft_wrapper.parameters(), lr = 3e-4)

    loss.backward()

    optim.step()

    sdft_wrapper.update_teacher_ema_()

def test_trainer():
    from torch.utils.data import Dataset
    from x_transformers import TransformerWrapper, Decoder

    class MockDataset(Dataset):
        def __init__(self, length = 10):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return "question", "answer"

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 512,
        attn_layers = Decoder(
            dim = 512,
            depth = 2
        )
    )

    def tokenizer_encode(prompts):
        return [torch.tensor([ord(c) for c in prompt]) for prompt in prompts]

    sdft = SDFT(
        model,
        student_max_response_len = 16,
        tokenizer_encode = tokenizer_encode,
    )

    from sdft_pytorch.sdft_pytorch import SDFTTrainer

    trainer = SDFTTrainer(
        sdft,
        dataset = MockDataset(),
        batch_size = 2,
        accelerate_kwargs = dict(cpu = True)
    )

    trainer()
