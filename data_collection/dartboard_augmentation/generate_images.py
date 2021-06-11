import json
import os
import random
import string

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from typing import Tuple, Iterator

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

def get_positions_in_square(square: int) -> Iterator[Tuple[float, float]]:
    square_image: Image.Image = Image.open(f"../perspective_shift/assets/{square}.png").convert("RGBA")
    try:
        square_image = square_image.filter(ImageFilter.MinFilter(7))
    except ValueError:
        pass
    choices = np.asarray(square_image)[:, :, -1]
    ys, xs = np.nonzero(choices)
    xs = xs / (square_image.width / 2) - 1
    ys = ys / (square_image.height / 2) - 1
    return zip(xs, ys)


def random_position(board):
    square = random.randint(0, 81)
    positions = get_positions_in_square(square)
    offset_positions = fix_offset(board, positions)
    position = random.choice(offset_positions)
    return position, square


def fix_offset(board: Image.Image, positions):
    """
    Turns the board relative positions to the actual board position.
    """

    MAGIC_NUM = 275  # Meme but it works

    result = []
    for x, y in positions:
        x = (x * MAGIC_NUM) + board.width / 2 - 2
        y = (y * MAGIC_NUM) + board.height / 2 + 3 # Better results
        result.append((x, y))
    return result


def display_markers(im: Image.Image, markers):
    marker: Image.Image = Image.open("../data_augmentation/assets/small_marker.png")
    for x, y in markers:
        im.paste(marker, (int(x)-marker.width//2, int(y)-marker.height//2), marker)
    im.show()


def load_data(folder: string):
    print(f"Loading from \"{folder}\"")
    data = []
    for file in os.listdir(folder):
        im = Image.open(f'./{folder}/{file}')
        data.append(im)
    print(f"Finished loading from \"{folder}\"")
    return data


def transform_darts(darts, dimensions):
    for dart in darts:
        dart.thumbnail(dimensions)


def generate(num_images: int):
    """
    Generate an image so it can be used as a training sample.
    """

    darts = load_data("darts")
    print(darts)
    boards = load_data("boards")
    DARTSIZE = 270

    for i in tqdm(range(num_images)):
        board = random.choice(boards).copy()
        im = Image.new('RGBA', board.size)
        positions = []
        squares = []
        for _ in range(1):
            dart = random.choice(darts)
            position, square = random_position(board)
            positions.append(position)
            squares.append(square)

            persp = np.asarray([[0, 0], [0, dart.size[1]//2], [dart.size[0]//2, dart.size[1]//2], [dart.size[0]//2, 0]], np.float32)
            # warp perspective
            # basic way: add gaussian noise to the 4 corner points
            warped_persp = persp.copy()
            for i in [0,1,3]:
                for j in range(2):
                    v = warped_persp[i][j]
                    v += random.gauss(0, 5)
                    # ensure none of the perspective points will fall outside the cropped image
                    # v = max(box[j] + 5, v)
                    # v = min(box[j + 2] - 5, v)
                    warped_persp[i][j] = v

            im_cv = cv2.cvtColor(np.array(dart), cv2.COLOR_RGBA2BGRA)
            matrix = cv2.getPerspectiveTransform(persp, warped_persp)
            warped_im = cv2.warpPerspective(im_cv, matrix, (dart.width, dart.height))
            warped_dart = Image.fromarray(cv2.cvtColor(warped_im, cv2.COLOR_BGRA2RGBA)).resize((DARTSIZE, DARTSIZE)).rotate(random.randint(0, 360), expand=True)

            im.paste(warped_dart, (int(position[0]) - warped_dart.width // 2, int(position[1]) - warped_dart.height // 2), warped_dart)

        persp = np.asarray([
            [im.size[0]//2 - 100, im.size[1]//2 - 100],
            [im.size[0]//2 - 100, im.size[1]//2 + 100],
            [im.size[0]//2 + 100, im.size[1]//2 + 100],
            [im.size[0]//2 + 100, im.size[1]//2 - 100]
        ], np.float32)
        # warp perspective
        # basic way: add gaussian noise to the 4 corner points
        warped_persp = persp.copy()
        for i in range(4):
            for j in range(2):
                v = warped_persp[i][j]
                v += random.gauss(0, 5)
                # ensure none of the perspective points will fall outside the cropped image
                # v = max(box[j] + 5, v)
                # v = min(box[j + 2] - 5, v)
                warped_persp[i][j] = v

        board.paste(im, (0, 0), im)
        im_cv = cv2.cvtColor(np.array(board), cv2.COLOR_RGB2BGR)
        matrix = cv2.getPerspectiveTransform(persp, warped_persp)
        warped_im = cv2.warpPerspective(im_cv, matrix, (board.width, board.height))
        warped_im = Image.fromarray(cv2.cvtColor(warped_im, cv2.COLOR_BGR2RGB))

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
        warped_im = warped_im.resize((256, 256), box=(
            warped_im.size[0]//2 - 450, warped_im.size[1]//2 - 450,
            warped_im.size[0]//2 + 450, warped_im.size[1]//2 + 450
        ))

        fname = f"{hex(random.randint(2**20, 2**24))[2:]}"
        warped_im.save(f"../../data/generated/{fname}.jpg")
        with open(f"../../data/generated/{fname}.json", "w") as write_file:
            data = {
                "square": squares,
                "position": positions
            }
            write_file.write(json.dumps(data))


if __name__ == '__main__':
    generate(10_000)
