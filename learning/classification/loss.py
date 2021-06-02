from PIL import Image
from torch.nn import Module
from scipy import ndimage
import numpy as np


class DistanceLoss(Module):
    def __init__(self):
        super().__init__()

        # precompute the centers of all 82 classes
        # for this, we use PNGs displayed in the labelling GUI
        center_cache = dict()
        for i in range(82):
            img: Image.Image = Image.open(f"data_collection/perspective_shift/assets/{i}.png").convert("RGBA")
            im_array = np.array(img)[:, :, -1]
            com = ndimage.center_of_mass(im_array)[::-1]
            center_cache[i] = com
