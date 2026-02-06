# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "einops",
#   "accelerate",
#   "tqdm",
#   "fire",
#   "x-mlps-pytorch",
#   "tensorboard",
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
from x_mlps_pytorch import MLP

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

# evaluation helper

@torch.no_grad()
def evaluate(model, dl, accelerator):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in dl:
        images = rearrange(images, 'b c d -> b (c d)')
        logits = model(images)
        preds = logits.argmax(dim = -1)
        
        preds, labels = accelerator.gather_for_metrics((preds, labels))
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return correct / total

# main training function

def train(
    num_stages = 10,
    epochs_per_stage = 1,
    batch_size = 64,
    lr = 1e-3,
    dim_hidden = 512,
    base_seed = 42
):
    accelerator = Accelerator(
        log_with = "tensorboard",
        project_dir = "./logs"
    )
    
    accelerator.init_trackers("permuted_mnist")
    
    # model and optimizer

    model = MLP(28 * 28, dim_hidden, dim_hidden, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # prepare model and optimizer once

    model, optimizer = accelerator.prepare(model, optimizer)
    
    all_perms = []
    global_step = 0
    
    for stage in range(num_stages):
        accelerator.print(f'\n--- Stage {stage} ---')
        
        # permutation for this stage

        seed = base_seed + stage
        perm = get_permutation(seed = seed)
        all_perms.append(perm)
        
        # dataset and dataloader for training

        train_ds = PermutedMNIST('./data', train = True, download = True, permutation = perm)
        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
        train_dl = accelerator.prepare(train_dl)
        
        # training loop for this stage

        model.train()
        
        for epoch in range(epochs_per_stage):
            pbar = tqdm(train_dl, desc = f'Stage {stage} Epoch {epoch}', disable = not accelerator.is_main_process)
            
            for images, labels in pbar:
                images = rearrange(images, 'b c d -> b (c d)')
                
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                accelerator.log({"loss": loss.item()}, step = global_step)
                pbar.set_postfix(loss = loss.item())

        # evaluation on all stages seen so far

        for i, prev_perm in enumerate(all_perms):
            test_ds = PermutedMNIST('./data', train = False, download = True, permutation = prev_perm)
            test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
            test_dl = accelerator.prepare(test_dl)
            
            accuracy = evaluate(model, test_dl, accelerator)
            accelerator.print(f'Accuracy Task {i}: {accuracy:.4f}')
            accelerator.log({f"accuracy/task_{i}": accuracy}, step = global_step)

    accelerator.end_training()
    accelerator.print('Training complete.')

if __name__ == '__main__':
    fire.Fire(train)
