import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(args.latent_size, args.hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.hidden_size * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(args.hidden_size * 8, args.hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.hidden_size * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(args.hidden_size * 4, args.hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.hidden_size * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(args.hidden_size * 2, args.hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.hidden_size),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(args.hidden_size, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(1, args.hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(args.hidden_size, args.hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(args.hidden_size * 2, args.hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(args.hidden_size * 4, args.hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(args.hidden_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)