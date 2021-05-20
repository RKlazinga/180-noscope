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

        self.cache = {}

        if is_test:
            self.images = self.images[-int(len(self.images) * test_fraction):]
        else:
            self.images = self.images[:-int(len(self.images) * test_fraction)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        name = self.images[idx]

        single_im = Image.open(f"../augmented_data/{name}")
        single = transforms.ToTensor()(single_im)
        with open(f"../augmented_data/{os.path.splitext(name)[0]}.json") as readfile:
            label = json.loads(readfile.read(-1))["perspective"]
            label = torch.tensor(label).flatten()
        ret_dict = {
            "input": single,
            "label": label
        }
        self.cache[idx] = ret_dict
        return ret_dict
