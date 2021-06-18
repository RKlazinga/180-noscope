import json
import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import numpy as np
from scipy import ndimage

from learning.perspective.network import warp_image_using_network


class ClassificationDataset(data.Dataset):

    # train, test, validation
    SPLIT = [0.8, 0.2, 0.0]

    def __init__(self, perspective_net, device, split_idx=0):
        super().__init__()

        self.perspective_net = perspective_net
        self.device = device

        # load all augmented CLASSIFICATION images, which can be used directly
        self.images = [("c", x) for x in os.listdir("data/generated") if not x.endswith((".keep", ".json", ".txt"))]

        # load all augmented PERSPECTIVE images, to be corrected later
        self.images.extend([("p", x) for x in os.listdir("data/augmented") if not x.endswith((".keep", ".json", ".txt"))])

        self.cache = {}

        cumulative = []
        for s in range(len(self.SPLIT)):
            cumulative.append(int(sum(self.SPLIT[:s]) * len(self.images)))
        cumulative.append(len(self.images))

        # extract correct fraction
        self.images = self.images[cumulative[split_idx]: cumulative[split_idx+1]]

        # precompute the centers of all 82 classes
        # for this, we use PNGs displayed in the labelling GUI
        self.center_cache = self.get_centers()

    @staticmethod
    def get_centers():
        center_cache = dict()
        for i in range(82):
            img: Image.Image = Image.open(f"data_collection/perspective_shift/assets/{i}.png").convert("RGBA")
            im_array = np.array(img)[:, :, -1]
            com = ndimage.center_of_mass(im_array)[::-1]
            com = com[0] / 460, com[1] / 460
            center_cache[i] = com

        # since the bullseye and the green ring around it will have approximately the same center of mass,
        # we shift one up and the other down slightly
        center_cache[0] = center_cache[0][0], center_cache[0][1] - 5
        center_cache[1] = center_cache[1][0], center_cache[1][1] + 5
        center_cache[82] = (0.0, 0.0)

        return center_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        im_type, name = self.images[idx]

        if im_type == "p":
            single_im = Image.open(f"data/augmented/{name}")
            warped_im = warp_image_using_network(self.perspective_net, self.device, single_im)

            with open(f"data/augmented/{os.path.splitext(name)[0]}.json") as readfile:
                info = json.loads(readfile.read(-1))
                label = list(info["darts"].values())

            try:
                label = label.index(1)
            except ValueError:
                label = 82

            ret_dict = {
                "input": transforms.ToTensor()(warped_im),
                "class_pos": torch.tensor(self.center_cache[label]).float()
            }
        elif im_type == "c":
            single_im = Image.open(f"data/generated/{name}")

            with open(f"data/generated/{os.path.splitext(name)[0]}.json") as readfile:
                info = json.loads(readfile.read(-1))
                label = info["square"][0]

            ret_dict = {
                "input": transforms.ToTensor()(single_im),
                "class_pos": torch.tensor(self.center_cache[label]).float()
            }

        else:
            raise ValueError(f"Unknown image type {im_type}")

        self.cache[idx] = ret_dict
        return ret_dict
