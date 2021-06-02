import os
import time

import torch
import torchsummary
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import learning.classification.dataset as dataset
import learning.perspective.eval_network as eval_network
from learning.util.conv_unit import ConvUnit


class ClassificationNetwork(nn.Module):

    IN_SIZE = 256
    CONV_LAYERS = 7
    CLASSES_OUT = 83  # 82 patches on a dartboard + no darts as the 83rd class

    def __init__(self, mult=8):
        """
        Set up a CNN to classify in which square the dart landed.

        :param mult: Size multiplier for all convolutional units.
        """
        super().__init__()

        self.layers = [
            ConvUnit(3, mult, 4, padding=2),
            nn.MaxPool2d(2)
        ]

        total_vars = mult
        for c in range(1, self.CONV_LAYERS):
            self.layers.append(ConvUnit(total_vars, total_vars * 2, 4, padding=2))
            total_vars *= 2
            self.layers.append(nn.MaxPool2d(2))

        self.layers.append(nn.Flatten())

        total_vars = int((self.IN_SIZE / 2**self.CONV_LAYERS)**2 * total_vars)
        self.layers.append(nn.Linear(total_vars, total_vars))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(total_vars, self.CLASSES_OUT))
        self.layers.append(nn.Softmax())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    epochs = 10

    network = ClassificationNetwork()

    if os.path.isfile("C_NETWORK.pth"):
        network.load_state_dict(torch.load("C_NETWORK.pth"))

    optimiser = optim.Adam(network.parameters())
    torchsummary.summary(network, input_size=(3, 256, 256))

    train_dataset = dataset.ClassificationDataset(split_idx=0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    test_dataset = dataset.ClassificationDataset(split_idx=1)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16)

    # ensure the train, test and validation dataset have no overlap!
    assert len(set(train_dataset.images).intersection(test_dataset.images)) == 0

    criterion = CrossEntropyLoss()

    prev_test_loss = 1e9

    start = time.time()
    for epoch in range(1, epochs+1):
        train_loss = 0
        for idx, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
            optimiser.zero_grad()

            out = network.forward(batch["input"])
            single_loss = criterion(out, batch["label"])

            single_loss.backward()
            optimiser.step()

            train_loss += single_loss.item()
        train_loss /= len(train_loader)

        test_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc=f"Testing epoch {epoch}")):
                out = network.forward(batch["input"])
                single_loss = criterion(out, batch["label"])
                test_loss += single_loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch}\t Train Loss={train_loss ** 0.5}\t Test Loss={test_loss ** 0.5}")

        if test_loss < prev_test_loss:
            torch.save(network.state_dict(), os.path.join(os.getcwd(), "C_NETWORK.pth"))
            prev_test_loss = test_loss

    print(f"Training took {time.time() - start} seconds")


