import torch
from sdft_pytorch import SDFT

def test_sdft():
    from x_transformers import TransformerWrapper, Decoder

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 512,
        attn_layers = Decoder(
            dim = 512,
            depth = 2
        )
    )

    sdft_wrapper = SDFT(model)
