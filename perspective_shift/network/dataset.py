import json
import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class PerspectiveDataset(data.Dataset):
    def __init__(self, test_fraction=0.2, is_test=False):
        super().__init__()
        self.images = [x for x in os.listdir("../augmented_data") if not x.endswith((".keep", ".json"))]
        if is_test:
            self.images = self.images[-int(len(self.images) * test_fraction):]
        else:
            self.images = self.images[:-int(len(self.images) * test_fraction)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        single_im = Image.open(f"../augmented_data/{name}")
        single = transforms.ToTensor()(single_im)
        with open(f"../augmented_data/{os.path.splitext(name)[0]}.json") as readfile:
            label = torch.tensor(json.loads(readfile.read(-1))["perspective"]).flatten()
            label = label / 256
        return {
            "input": single,
            "label": label
        }