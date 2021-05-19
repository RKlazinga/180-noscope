import json
import os
import random

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from tqdm import tqdm


def display_markers(im: Image.Image, markers):
    marker: Image.Image = Image.open("data_augmentation/assets/small_marker.png")
    for x, y in markers:
        im.paste(marker, (int(x)-marker.width//2, int(y)-marker.height//2), marker)
    im.show()


def augment_all():
    skip_list = []
    if os.path.isfile("corrected_data/_skip.txt"):
        with open("corrected_data/_skip.txt", "r") as readfile:
            skip_list = readfile.read(-1).split("|")

    for i in tqdm(os.listdir("raw_data")):
        if i == ".keep" or i in skip_list:
            continue
        for _ in range(10):
            augment(i)


def augment(im_path):
    """
    Modify an image so it can be used as an additional training sample.

    :param im_path: Name of the image file in raw_data. Note that it must have been labelled already!
    """
    # change directory to toplevel of repo (parent of data_augmentation)
    os.chdir(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

    im_name, im_ext = os.path.splitext(im_path)
    if im_path not in os.listdir("raw_data"):
        raise FileNotFoundError(f"{im_path} could not be found in the list of raw images")

    if im_name + ".json" not in os.listdir("corrected_data"):
        raise FileNotFoundError(f"{im_name} has not been labelled yet! (no file '{im_name}.json' in corrected_data)")

    with open(f"corrected_data/{im_name}.json") as read_file:
        im_label = json.loads(read_file.read(-1))
    persp = np.float32(im_label["perspective"])

    im: Image.Image = Image.open(f"raw_data/{im_path}")
    # downscale image to reasonable height
    scale_factor = 500 / im.height
    persp = persp * scale_factor
    im.thumbnail([1000000, 500])
    im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    # determine crop box
    crop_amount = (im.width - 500)
    left_crop = random.randint(crop_amount//4, 3 * crop_amount // 4)
    # left_crop = crop_amount//2
    right_crop = crop_amount - left_crop
    box = [
        left_crop,
        0,
        im.width - right_crop,
        im.height
    ]

    # warp perspective
    # basic way: add gaussian noise to the 4 corner points
    warped_persp = persp.copy()
    for i in range(4):
        for j in range(2):
            v = warped_persp[i][j]
            v += random.gauss(0, 5)
            # ensure none of the perspective points will fall outside the cropped image
            v = max(box[j] + 5, v)
            v = min(box[j+2] - 5, v)
            warped_persp[i][j] = v

    matrix = cv2.getPerspectiveTransform(persp, warped_persp)
    warped_im = cv2.warpPerspective(im_cv, matrix, (im.width, im.height))
    warped_im = Image.fromarray(cv2.cvtColor(warped_im, cv2.COLOR_BGR2RGB))

    # run crop on warped image
    warped_im = warped_im.crop(box)
    # adjust warped coordinates according to crop
    for i in range(4):
        warped_persp[i][0] -= box[0]
        warped_persp[i][1] -= box[1]

    # adjust image colour balance, saturation and contrast
    warped_im = ImageEnhance.Color(warped_im).enhance(1 + random.randint(-10, 10)/40)
    warped_im = ImageEnhance.Contrast(warped_im).enhance(1 + random.randint(-10, 10)/40)
    warped_im = ImageEnhance.Brightness(warped_im).enhance(1 + random.randint(-10, 10)/40)

    # add noise
    noise_strength = random.uniform(5, 15)
    warped_im_arr = np.float64(np.array(warped_im))
    warped_im_arr += np.random.normal(0, noise_strength, warped_im_arr.shape)
    warped_im_arr = np.clip(warped_im_arr, 0, 255)
    warped_im = Image.fromarray(np.uint8(warped_im_arr))

    fname = f"{im_name}-{hex(random.randint(2**20, 2**24))[2:]}"
    warped_im.save(f"augmented_data/{fname}{im_ext}")
    with open(f"augmented_data/{fname}.json", "w") as write_file:
        data = {
            "darts": im_label["darts"],
            "perspective": warped_persp.tolist()
        }
        write_file.write(json.dumps(data))
    return warped_im, warped_persp


if __name__ == '__main__':
    augment_all()
    # w_im, w_p = augment("01.jpg")
    # display_markers(w_im, w_p)