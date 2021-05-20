import json
import os

import torch
from PIL import Image
from torchvision import transforms

from data_augmentation.augment import display_markers
from perspective_shift.network.network import PerspectiveNetwork

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.getcwd()))

    paths = [x for x in os.listdir("augmented_data")[-100::24] if not x.endswith(".json")]
    network = PerspectiveNetwork()
    network.load_state_dict(torch.load("perspective_shift/NETWORK.pth"))

    for path in paths:
        im = Image.open(f"augmented_data/{path}")

        with open(f"augmented_data/{os.path.splitext(path)[0]}.json") as readfile:
            label = json.loads(readfile.read(-1))["perspective"]

        out = network.forward(transforms.ToTensor()(im).view(1, 3, 256, 256))

        out_parsed = []
        for i in range(4):
            out_parsed.append([out[0][i*2], out[0][i*2 + 1]])
        display_markers(im, label, out_parsed)