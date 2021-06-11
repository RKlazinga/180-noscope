import torch
from PIL import Image
from torch.nn import Module, CrossEntropyLoss
from scipy import ndimage
import numpy as np

from learning.classification.dataset import ClassificationDataset


class DistanceLoss(Module):
    def __init__(self):
        super().__init__()

        # precompute the centers of all 82 classes
        # for this, we use PNGs displayed in the labelling GUI
        self.center_cache = ClassificationDataset.get_centers()

    @staticmethod
    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / 255

    def forward(self, pred, true):
        """
        :param pred: Predicted class
        :param true: True class
        :return: Distance loss between classes
        """
        total = 0
        for i in range(pred.shape[0]):
            total += self.dist(self.center_cache[torch.argmax(pred[i]).item()],
                               self.center_cache[true[i].item()])
        return total


class CombinedLoss(Module):
    def __init__(self, distance_weight=1):
        super().__init__()

        self.ce_loss = CrossEntropyLoss()
        self.dist_loss = DistanceLoss()
        self.dist_weight = distance_weight

    def forward(self, pred, true):
        return self.dist_weight * self.dist_loss(pred, true) + (1 - self.dist_weight) * self.ce_loss(pred, true)

