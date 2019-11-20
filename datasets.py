import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import torch
import scipy.misc

class vae_dataset(data.Dataset):

    def __init__(self, args):
    
        self.minst = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        
        self.args = args

    def __len__(self):
        return len(self.minst)

    def __getitem__(self, index):
    
        # return img   : N*1*Dx
        # return noise : N*L*Dz
        img = self.minst[index][0]
        noise = torch.randn(self.args.sample_times, self.args.latent_dim)
        
        img = img.reshape(1, -1)
        
        return img, noise


