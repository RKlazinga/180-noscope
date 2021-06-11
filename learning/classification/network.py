import os
import time

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

import learning.classification.dataset as dataset
from learning.perspective.network import PerspectiveNetwork
from learning.util.conv_unit import ConvUnit


class ClassificationNetwork(nn.Module):

    IN_SIZE = 256
    CONV_LAYERS = 8
    # CLASSES_OUT = 83  # 82 patches on a dartboard + no darts as the 83rd class

    def __init__(self, mult=8):
        """
        Set up a CNN to classify in which square the dart landed.

        :param mult: Size multiplier for all convolutional units.
        """
        super().__init__()

        self.layers = [
            ConvUnit(5, mult, 4, padding=2),
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
        self.layers.append(nn.Linear(total_vars, 2))  # self.CLASSES_OUT))
        self.layers.append(nn.Sigmoid())
        # self.layers.append(nn.Softmax())

        self.model = nn.Sequential(*self.layers)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    epochs = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-6 * BATCH_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perspective_network = PerspectiveNetwork(mult=8).to(device)
    perspective_network.load_state_dict(torch.load("NETWORK-2.71.pth", map_location=device))

    network = ClassificationNetwork(32).to(device)

    #if os.path.isfile("C2_NETWORK.pth"):
    #    network.load_state_dict(torch.load("C2_NETWORK.pth"))

    optimiser = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    train_dataset = dataset.ClassificationDataset(perspective_network, device, split_idx=0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    test_dataset = dataset.ClassificationDataset(perspective_network, device, split_idx=1)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # ensure the train, test and validation dataset have no overlap!
    assert len(set(train_dataset.images).intersection(test_dataset.images)) == 0

    criterion = MSELoss()

    prev_test_loss = 1e9

    start = time.time()
    for epoch in range(1, epochs+1):
        train_loss = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            optimiser.zero_grad()
            batch["input"] = batch["input"].to(device)
            batch["class_pos"] = batch["class_pos"].to(device)

            out = network.forward(batch["input"])
            #print(out, batch["class_pos"], criterion(out, batch["class_pos"]))
            single_loss = criterion(out, batch["class_pos"])

            single_loss.backward()
            optimiser.step()

            train_loss += single_loss.item()
        train_loss /= len(train_loader)

        test_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader)):
                batch["input"] = batch["input"].to(device)
                batch["class_pos"] = batch["class_pos"].to(device)
                out = network.forward(batch["input"])
                single_loss = criterion(out, batch["class_pos"])
                test_loss += single_loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch}\t Train Loss={train_loss**0.5}\t Test Loss={test_loss**0.5}")

        if test_loss < prev_test_loss:
            torch.save(network.state_dict(), os.path.join(os.getcwd(), "C2_NETWORK.pth"))
            prev_test_loss = test_loss

    print(f"Training took {time.time() - start} seconds")


