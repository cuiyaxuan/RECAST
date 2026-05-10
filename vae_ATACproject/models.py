import torch
import torch.nn as nn


class DS(torch.utils.data.Dataset):
    def __init__(self, X, c):
        self.X = torch.tensor(X).float()
        self.c = c

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.c[i]


class VAE(nn.Module):
    def __init__(self, dim: int, classes: int, z: int = 64):
        super().__init__()
        self.emb = nn.Embedding(classes, 16)

        self.enc = nn.Sequential(
            nn.Linear(dim + 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, z)
        self.lv = nn.Linear(256, z)

        self.dec = nn.Sequential(
            nn.Linear(z + 16, 512),
            nn.ReLU(),
            nn.Linear(512, dim),
            nn.Softplus(),
        )

    def encode(self, x, c):
        h = torch.cat([x, self.emb(c)], dim=1)
        h = self.enc(h)
        return self.mu(h), self.lv(h)

    def reparam(self, mu, lv):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * lv)

    def decode(self, z, c):
        return self.dec(torch.cat([z, self.emb(c)], dim=1))

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        z = self.reparam(mu, lv)
        return self.decode(z, c), mu, lv, z


class Projector(nn.Module):
    def __init__(self, z: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, z),
        )

    def forward(self, x):
        return self.net(x)