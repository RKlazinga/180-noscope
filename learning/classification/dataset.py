import json
import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

import learning.classification.network as classification_network


class ClassificationDataset(data.Dataset):

    # train, test, validation
    SPLIT = [0.8, 0.2, 0.0]

    def __init__(self, split_idx=0):
        super().__init__()
        self.images = [x for x in os.listdir("data/corrected") if not x.endswith((".keep", ".json", ".txt"))]

        self.cache = {}

        cumulative = []
        for s in range(len(self.SPLIT)):
            cumulative.append(int(sum(self.SPLIT[:s]) * len(self.images)))
        cumulative.append(len(self.images))

        # extract correct fraction
        self.images = self.images[cumulative[split_idx]: cumulative[split_idx+1]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        name = self.images[idx]

        single_im = Image.open(f"data/corrected/{name}")

        # resize to correct dimensions for network
        single_im = single_im.resize((classification_network.ClassificationNetwork.IN_SIZE,
                                      classification_network.ClassificationNetwork.IN_SIZE))

        single = transforms.ToTensor()(single_im)

        with open(f"data/corrected/{os.path.splitext(name)[0]}.json") as readfile:
            info = json.loads(readfile.read(-1))
            label = list(info["darts"].values())

        # IMPORTANT: CrossEntropyLoss requires a single number as the label, namely the correct index
        # Hence we reduce label down to a single index, even if there are multiple darts(!)
        try:
            label = label.index(1)
        except ValueError:
            label = 82

        ret_dict = {
            "input": single,
            "label": label,

        }
        self.cache[idx] = ret_dict
        return ret_dict
