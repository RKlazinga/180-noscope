import json
import os
import random
import string

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from tqdm import tqdm
from typing import Tuple, Iterator


def get_positions_in_square(square: int) -> Iterator[Tuple[float, float]]:
    square_image: Image.Image = Image.open(f"../perspective_shift/assets/{square}.png")
    choices = np.asarray(square_image)[:, :, -1]
    ys, xs = np.nonzero(choices)
    xs = (xs - square_image.width / 2) / square_image.width / 2
    ys = (ys - square_image.width / 2) / square_image.width / 2
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

    # TODO: Perhaps add a feather operation to get less close to the border

    MAGIC_NUM = 420  # Meme but it works

    result = []
    for x, y in positions:
        x = (x * MAGIC_NUM) + board.width / 2
        y = (y * MAGIC_NUM) + board.height / 2 + 2  # Better results
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
    transform_darts(darts, (100, 100))
    print(darts)
    boards = load_data("boards")

    for i in tqdm(range(num_images)):
        board = random.choice(boards).copy()
        im = Image.new('RGBA', board.size)
        positions = []
        squares = []
        for _ in range(3):
            dart = random.choice(darts)
            position, square = random_position(board)
            positions.append(position)
            squares.append(square)

            persp = np.asarray([[0, 0], [0, dart.size[1]], [dart.size[0], dart.size[1]], [dart.size[0], 0]], np.float32)
            # warp perspective
            # basic way: add gaussian noise to the 4 corner points
            warped_persp = persp.copy()
            for i in range(4):
                for j in range(2):
                    v = warped_persp[i][j]
                    v += random.gauss(0, 20)
                    # ensure none of the perspective points will fall outside the cropped image
                    # v = max(box[j] + 5, v)
                    # v = min(box[j + 2] - 5, v)
                    warped_persp[i][j] = v

            im_cv = cv2.cvtColor(np.array(dart), cv2.COLOR_RGBA2BGRA)
            matrix = cv2.getPerspectiveTransform(persp, warped_persp)
            warped_im = cv2.warpPerspective(im_cv, matrix, (dart.width, dart.height))
            warped_dart = Image.fromarray(cv2.cvtColor(warped_im, cv2.COLOR_BGRA2RGBA))

            im.paste(dart, (int(position[0]) - dart.width // 2, int(position[1]) - dart.height // 2), dart)

        board.paste(im, (0, 0), im)
        #board.show()
        fname = f"{hex(random.randint(2**20, 2**24))[2:]}"
        board.save(f"../../data/generated/{fname}.jpg")
        with open(f"../../data/generated/{fname}.json", "w") as write_file:
            data = {
                "square": squares,
                "position": positions
            }
            write_file.write(json.dumps(data))


if __name__ == '__main__':
    generate(5)
