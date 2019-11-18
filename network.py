import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(args.img_size*args.img_size, 128)
        self.fc_mu = nn.Linear(128, args.latent_dim)
        self.fc_sig = nn.Linear(128, args.latent_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        sig = torch.exp(self.fc_sig(x))
        return mu, sig


class decoder(nn.Module):
    def __init__(self, args):
        super(decoder, self).__init__()
        self.fc1 = nn.Linear(args.latent_dim, 128)
        self.fc_mu = nn.Linear(128, args.img_size*args.img_size)
        self.fc_sig = nn.Linear(128, args.img_size*args.img_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        sig = torch.exp(self.fc_sig(x))
        return mu, sig



