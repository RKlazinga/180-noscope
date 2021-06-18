import json
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchsummary

from data_collection.augmentation.augment import display_markers
import learning.perspective.network as perspective_network


def eval_network(network, paths):
    with torch.no_grad():
        combined_im = Image.new("RGB", (256 * len(paths), 256))
        for idx, path in enumerate(paths):
            im = Image.open(f"data/augmented/{path}")

            with open(f"data/augmented/{os.path.splitext(path)[0]}.json") as readfile:
                label = json.loads(readfile.read(-1))["perspective"]

            out = network.forward(transforms.ToTensor()(im).view(1, 3, 256, 256))

            out_parsed = []
            for i in range(4):
                out_parsed.append([out[0][i*2], out[0][i*2 + 1]])
            marked_im = display_markers(im, label, out_parsed)
            # warped_im = perspective_network.warp_image_from_coords(np.float32(out_parsed), im)

            combined_im.paste(marked_im, (idx * 256, 0))
            # combined_im.paste(warped_im, (idx * 256, 256))
        combined_im.show()
        combined_im.save("temp.png")



if __name__ == '__main__':

    network = perspective_network.PerspectiveNetwork()
    network.load_state_dict(torch.load("NETWORK-2.71.pth"))
    torchsummary.summary(network, (3, 256, 256))
    paths = [x for x in os.listdir("data/augmented")[-100::24] if not x.endswith(".json")]

    eval_network(network, paths)