# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "einops",
#   "accelerate",
#   "tqdm",
#   "fire",
# ]
# ///

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from accelerate import Accelerator
from einops import rearrange
from tqdm import tqdm
import fire

# simple fixed random permutation

def get_permutation(seed = 42):
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(28 * 28, generator = generator)

# dataset with permutation

class PermutedMNIST(datasets.MNIST):
    def __init__(self, *args, permutation = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation = permutation

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image = transforms.ToTensor()(image)
        image = rearrange(image, 'c h w -> c (h w)')
        
        if self.permutation is not None:
            image = image[..., self.permutation]
            
        return image, label

# simple mlp

def MLP(dim_in, dim_out, dim_hidden = 512):
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_out)
    )

# main training function

def train(
    epochs = 1,
    batch_size = 64,
    lr = 1e-3,
    dim_hidden = 512,
    seed = 42
):
    accelerator = Accelerator()
    
    # model and optimizer

    model = MLP(28 * 28, 10, dim_hidden = dim_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # permutation

    perm = get_permutation(seed = seed)
    
    # dataset and dataloader

    train_ds = PermutedMNIST('./data', train = True, download = True, permutation = perm)
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    
    # prepare for acceleration

    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    
    # training loop

    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(train_dl, desc = f'Epoch {epoch}', disable = not accelerator.is_main_process)
        
        for images, labels in pbar:
            images = rearrange(images, 'b c d -> b (c d)')
            
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_postfix(loss = loss.item())

    accelerator.print('Training complete.')

if __name__ == '__main__':
    fire.Fire(train)
