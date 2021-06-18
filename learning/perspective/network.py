import os
import time

import cv2
import torch
import torchsummary
from PIL import Image
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from learning.perspective.dataset import PerspectiveDataset
import learning.perspective.eval_network as eval_network
from learning.util.conv_unit import ConvUnit


class PerspectiveNetwork(nn.Module):
    def __init__(self, mult=8):
        """
        Set up a CNN to find the dart board and correct for perspective.

        :param mult: Size multiplier for all convolutional units.
        """
        super().__init__()

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


def warp_image_using_network(p_network, device, image):
    single = transforms.ToTensor()(image).view(1, 3, 256, 256)

    # feed through perspective network
    with torch.no_grad():
        perspective_out = p_network(single.to(device))

    out_parsed = []
    for i in range(4):
        out_parsed.append([perspective_out[0][i*2], perspective_out[0][i*2 + 1]])
    perspective_in = np.float32(out_parsed)

    return warp_image_from_coords(perspective_in, image)

def warp_image_from_coords(perspective_in, im):
    # apply perspective network answer
    im_cv = np.array(im)

    radius = 80
    padding = 128 - radius
    # offset is the distance between the center-line and the corner we are targeting
    # on 389 radius the offset is 63, so
    offset = radius * 63/389

    size = 2 * padding + 2 * radius

    perspective_target = np.float32([
        [padding + radius - offset, padding],
        [padding + radius - offset, padding + 2*radius],
        [padding, padding + radius - offset],
        [padding + 2*radius, padding + radius - offset]
    ])

    matrix = cv2.getPerspectiveTransform(perspective_in, perspective_target)
    result = cv2.warpPerspective(im_cv, matrix, (size, size))
    return Image.fromarray(result)


if __name__ == '__main__':

    writer = SummaryWriter()

    epochs = 10

    network = PerspectiveNetwork()

    if os.path.isfile("NETWORK.pth"):
        network.load_state_dict(torch.load("NETWORK.pth"))

    optimiser = optim.Adam(network.parameters())
    torchsummary.summary(network, input_size=(3, 256, 256))

    train_dataset = PerspectiveDataset(split_idx=0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    test_dataset = PerspectiveDataset(split_idx=1)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)

    validation_dataset = PerspectiveDataset(split_idx=2)
    validation_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)

    # ensure the train, test and validation dataset have no overlap!
    assert len(set(train_dataset.images).intersection(test_dataset.images)) == 0
    assert len(set(test_dataset.images).intersection(validation_dataset.images)) == 0
    assert len(set(train_dataset.images).intersection(validation_dataset.images)) == 0

    assert len(train_dataset) + len(test_dataset) + len(validation_dataset) == \
           len([x for x in os.listdir("data/augmented") if not x.endswith((".keep", ".json"))])

    # ensure the overlap of ground images between datasets is minimal
    assert len(set([x.split("-")[0] for x in train_dataset.images]).intersection(set([x.split("-")[0] for x in test_dataset.images]))) < 2

    criterion = MSELoss()

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

            writer.add_scalar("Train Accuracy", single_loss.item(), epoch * len(train_loader) + idx)

            train_loss += single_loss.item()
        train_loss /= len(train_loader)

        test_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc=f"Testing epoch {epoch}")):
                out = network.forward(batch["input"])
                single_loss = criterion(out, batch["label"])
                writer.add_scalar("Test Accuracy", single_loss.item(), epoch * len(test_loader) + idx)
                test_loss += single_loss.item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch}\t Train Loss={train_loss ** 0.5}\t Test Loss={test_loss ** 0.5}")

        if test_loss < prev_test_loss:
            torch.save(network.state_dict(), os.path.join(os.getcwd(), "NETWORK.pth"))
            prev_test_loss = test_loss

    # validate
    validation_loss = 0
    with torch.no_grad():
        for batch in tqdm(validation_loader):
            out = network.forward(batch["input"])
            single_loss = criterion(out, batch["label"])
            validation_loss += single_loss.item()
    validation_loss /= len(validation_loader)

    print(f"Training took {time.time() - start} seconds")
    print(f"Validation loss after {epochs}: {validation_loss ** 0.5}")
    print("Displaying some examples")
    eval_network.eval_network(network, validation_dataset.images[:3])


