# apply both the perspective and classification network to an image
import math

import cv2
import torch
from PIL import Image, ImageFont, ImageDraw
from scipy import ndimage
from torchvision import transforms
import numpy as np

from learning.classification.network import ClassificationNetwork
from learning.perspective.network import PerspectiveNetwork, warp_image
import os
import random


def classify_raw(im_path, device, pnet, cnet):
    im: Image.Image = Image.open(im_path).convert("RGB")

    # crop image down to square and resize to 256
    if im.width > im.height:
        box = ((im.width - im.height)/2, 0, im.width - (im.width - im.height)/2, im.height)
    else:
        box = (0, (im.height - im.width)/2, im.width, im.height - (im.height - im.width)/2)
    im = im.resize((256, 256), box=box)

    warped_im = warp_image(pnet, device, im)

    u = torch.linspace(0, 1, 256)
    u = u.repeat(256, 1)
    v = u.transpose(0, 1)
    uv = torch.stack([u, v], dim=0)

    classification_in_tensor = torch.cat([transforms.ToTensor()(warped_im), uv], dim=0).view(1, 5, 256, 256).to(device)
    c_out = cnet(classification_in_tensor)[0]
    print(c_out)

    # find nearest class
    classification_choice = -1
    closest_dist = 1e9
    for i in range(82):
        img: Image.Image = Image.open(f"data_collection/perspective_shift/assets/{i}.png").convert("RGBA")
        im_array = np.array(img)[:, :, -1]
        com = ndimage.center_of_mass(im_array)[::-1]
        com_x = com[0] / 460
        com_y = com[1] / 460

        dist = math.sqrt((c_out[0] - com_x) ** 2 + (c_out[1] - com_y) ** 2)
        if dist < closest_dist:
            closest_dist = dist
            classification_choice = i

    # classification_choice = torch.argmax(classification_out)

    im_show = Image.new("RGB", (3 * 256, 300))
    font = ImageFont.FreeTypeFont(r"C:\Windows\fonts\arial.ttf", size=22)
    im_draw = ImageDraw.Draw(im_show)

    # add text
    im_draw.text((10, 266), "Input image", font=font)
    im_draw.text((266, 266), "Perspective warped", font=font)
    im_draw.text((522, 266), "Classified region", font=font)

    im_show.paste(im, (0, 0))
    im_show.paste(warped_im, (256, 0))

    label_im = Image.open(f"data_collection/perspective_shift/assets/{int(classification_choice)}.png")
    label_im = label_im.resize((256, 256))
    im_show.paste(label_im, (512, 0))
    im_draw.ellipse((512 + 256 * c_out[0] - 3,
                     256 * c_out[1] - 3,
                     512 + 256 * c_out[0] + 3,
                     256 * c_out[1] + 3))

    im_show.show()


if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perspective_net = PerspectiveNetwork().to(dev)
    perspective_net.load_state_dict(torch.load("NETWORK-2.71.pth"))

    classification_net = ClassificationNetwork().to(dev)
    classification_net.load_state_dict(torch.load("C2_NETWORK.pth"))\

    classify_raw(random.choice([f"data/generated/{file}" for file in os.listdir("data/generated/") if not file.endswith('json')]), dev, perspective_net, classification_net)
