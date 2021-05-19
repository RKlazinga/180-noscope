import os

import torch
from torchvision import transforms
from PIL import Image
from torch import nn


class ConvUnit(nn.Module):
    """
    Basic combination of convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__()

        self.steps = [
            nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        ]

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)


class PerspectiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        mult = 16
        in_size = 256
        conv_layers = 6
        regression_out = 8

        self.layers = [
            ConvUnit(3, mult, 4, padding=2),
            nn.MaxPool2d(2)
        ]

        total_vars = mult
        for c in range(1, conv_layers):
            self.layers.append(ConvUnit(total_vars, total_vars * 2, 4, padding=2))
            total_vars *= 2
            self.layers.append(nn.MaxPool2d(2))

        self.layers.append(nn.Flatten())

        total_vars = int((in_size / 2**conv_layers)**2 * total_vars)
        self.layers.append(nn.Linear(total_vars, total_vars // 2))
        self.layers.append(nn.Linear(total_vars // 2, regression_out))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    network = PerspectiveNetwork()

    single_im = Image.open("../augmented_data/IMG_20210511_142817727-1f47bf.jpg")
    single = transforms.ToTensor()(single_im).view(1, 3, 256, 256)
    print(network.forward(single))
