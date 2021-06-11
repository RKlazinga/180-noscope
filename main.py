# apply both the perspective and classification network to an image
import cv2
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
import numpy as np

from learning.classification.network import ClassificationNetwork
from learning.perspective.network import PerspectiveNetwork


def classify_raw(im_path, pnet, cnet):
    im: Image.Image = Image.open(im_path).convert("RGB")

    # crop image down to square and resize to 256
    if im.width > im.height:
        box = ((im.width - im.height)/2, 0, im.width - (im.width - im.height)/2, im.height)
    else:
        box = (0, (im.height - im.width)/2, im.width, im.height - (im.height - im.width)/2)
    im = im.resize((256, 256), box=box)

    out = pnet(transforms.ToTensor()(im).view(1, 3, 256, 256))

    out_parsed = []
    for i in range(4):
        out_parsed.append([out[0][i*2], out[0][i*2 + 1]])

    im_cv = np.array(im)

    radius = 80
    padding = 128 - radius
    # offset is the distance between the center-line and the corner we are targeting
    # on 389 radius the offset is 63, so
    offset = radius * 63/389

    size = 2 * padding + 2 * radius
    print(size)
    perspective_in = np.float32(out_parsed)
    perspective_target = np.float32([
        [padding + radius + offset, padding],
        [padding + radius - offset, padding + 2*radius],
        [padding, padding + radius - offset],
        [padding + 2*radius, padding + radius + offset]
    ])

    matrix = cv2.getPerspectiveTransform(perspective_in, perspective_target)
    result = cv2.warpPerspective(im_cv, matrix, (size, size))
    warped_im = Image.fromarray(result)

    classification_in_tensor = transforms.ToTensor()(warped_im).view(1, 3, 256, 256)
    classification_out = cnet(classification_in_tensor)

    classification_choice = torch.argmax(classification_out)

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

    im_show.show()



if __name__ == '__main__':
    perspective_net = PerspectiveNetwork()
    perspective_net.load_state_dict(torch.load("NETWORK-2.71.pth"))

    classification_net = ClassificationNetwork()
    classification_net.load_state_dict(torch.load("C_NETWORK.pth"))\

    classify_raw("data/raw/IMG_20210511_142817727.jpg", perspective_net, classification_net)
