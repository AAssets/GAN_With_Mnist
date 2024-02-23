import torch
import torch.nn as nn
import argparse
import numpy as np


class Generator(nn.Module):

    def __init__(self,args):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, args.image_size* args.image_size),
            nn.Sigmoid(),
        )

        self.image_size= args.image_size

    def forward(self, z):

        output = self.model(z)
        image = output.reshape(z.shape[0], 1, self.image_size, self.image_size)

        return image


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(args.image_size*args.image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):

        prob = self.model(image.reshape(image.shape[0], -1))

        return prob

