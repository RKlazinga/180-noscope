import json
import os
import random

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from tqdm import tqdm


KELVIN_TABLE = [
    (255, 196, 137),
    (255, 209, 163),
    (255, 219, 186),
    (255, 228, 206),
    (255, 236, 224),
    (255, 243, 239),
    (255, 249, 253),
    (245, 243, 255),
    (235, 238, 255),
]


def display_markers(im: Image.Image, true_markers, output_markers=None):
    im = im.copy()
    marker: Image.Image = Image.open("data_collection/augmentation/assets/small_marker.png")
    for x, y in true_markers:
        im.paste(marker, (int(x)-marker.width//2, int(y)-marker.height//2), marker)
    if output_markers:
        red_marker: Image.Image = Image.open("data_collection/augmentation/assets/red_marker.png")
        for x, y in output_markers:
            im.paste(red_marker, (int(x)-red_marker.width//2, int(y)-red_marker.height//2), red_marker)
    return im


def augment_all():
    skip_list = []
    if os.path.isfile("data/corrected/_skip.txt"):
        with open("data/corrected/_skip.txt", "r") as readfile:
            skip_list = readfile.read(-1).split("|")

    for i in tqdm(os.listdir("data/raw")):
        if i == ".keep" or i in skip_list:
            continue
        for _ in range(20):
            augment(i)


def augment(im_path):
    """
    Modify an image so it can be used as an additional training sample.

    :param im_path: Name of the image file in raw. Note that it must have been labelled already!
    """
    # change directory to toplevel of repo (parent of augmentation)
    os.chdir(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

    im_name, im_ext = os.path.splitext(im_path)
    if im_path not in os.listdir("data/raw"):
        raise FileNotFoundError(f"{im_path} could not be found in the list of raw images")

    if im_name + ".json" not in os.listdir("data/corrected"):
        raise FileNotFoundError(f"{im_name} has not been labelled yet! (no file '{im_name}.json' in corrected)")

    with open(f"data/corrected/{im_name}.json") as read_file:
        im_label = json.loads(read_file.read(-1))
    persp = np.float32(im_label["perspective"])

    im: Image.Image = Image.open(f"data/raw/{im_path}")
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

    # scale down to final size
    warped_im = warped_im.resize((256, 256))
    for i in range(4):
        warped_persp[i][0] *= 256 / 500
        warped_persp[i][1] *= 256 / 500

    # adjust image colour balance, saturation and contrast
    warped_im = ImageEnhance.Color(warped_im).enhance(random.uniform(0.9, 1.2))
    warped_im = ImageEnhance.Contrast(warped_im).enhance(random.uniform(0.8, 1.2))
    warped_im = ImageEnhance.Brightness(warped_im).enhance(random.uniform(0.8, 1.2))

    # adjust image temperature
    # thanks to Mark Ransom (https://stackoverflow.com/a/11888449)
    temp_r, temp_g, temp_b = random.choice(KELVIN_TABLE)
    convert_matrix = (temp_r / 255.0, 0.0, 0.0, 0.0,
                      0.0, temp_g / 255.0, 0.0, 0.0,
                      0.0, 0.0, temp_b / 255.0, 0.0)
    warped_im = warped_im.convert("RGB", convert_matrix)

    # add noise
    noise_strength = random.uniform(5, 10)
    warped_im_arr = np.float64(np.array(warped_im))
    warped_im_arr += np.random.normal(0, noise_strength, warped_im_arr.shape)
    warped_im_arr = np.clip(warped_im_arr, 0, 255)
    warped_im = Image.fromarray(np.uint8(warped_im_arr))

    fname = f"{im_name}-{hex(random.randint(2**20, 2**24))[2:]}"
    warped_im.save(f"data/augmented/{fname}{im_ext}")
    with open(f"data/augmented/{fname}.json", "w") as write_file:
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