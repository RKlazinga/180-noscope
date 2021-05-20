import os

import torch
import torchsummary
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn, optim
from tqdm import tqdm

from perspective_shift.network.dataset import PerspectiveDataset


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

        mult = 8
        in_size = 256
        conv_layers = 7
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
        self.layers.append(nn.Linear(total_vars, total_vars))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(total_vars, regression_out))
        self.layers.append(nn.ReLU())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    network = PerspectiveNetwork()
    optimiser = optim.Adam(network.parameters())
    torchsummary.summary(network, input_size=(3, 256, 256))

    train_dataset = PerspectiveDataset()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    test_dataset = PerspectiveDataset(is_test=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)

    # ensure the train and test dataset have no overlap!
    assert len(set(train_dataset.images).intersection(test_dataset.images)) == 0

    criterion = MSELoss()

    for epoch in range(10):
        train_loss = 0
        for batch in tqdm(train_loader):
            optimiser.zero_grad()

            out = network.forward(batch["input"])

            single_loss = criterion(out, batch["label"])

            single_loss.backward()
            optimiser.step()

            train_loss += single_loss.item()
        train_loss /= len(train_loader)

        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                out = network.forward(batch["input"])
                single_loss = criterion(out, batch["label"])
                test_loss += single_loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch}\t Train Loss={train_loss ** 0.5}\t Test Loss={test_loss ** 0.5}")
        torch.save(network.state_dict(), os.path.join(os.getcwd(),"NETWORK.pth"))


