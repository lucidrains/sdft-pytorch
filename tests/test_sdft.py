import torch
from sdft_pytorch.sdft_pytorch import SDFT

def test_sdft():
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
        tokenizer_encode = tokenizer_encode,
    )

    loss = sdft_wrapper(
        questions = ['12+48', '2*3'],
        answers = ['60', '6']
    )

    loss.backward()
