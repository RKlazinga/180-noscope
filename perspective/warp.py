# import necessary libraries

import cv2
import numpy as np


def left_top_width_height(l, t, w, h):
    return np.float32([
        [l, t],
        [l, t+h],
        [l+w, t],
        [l+w, t+h]
    ])


while True:
    im = cv2.imread("test.jpg")

    # Locate points of the documents or object which you want to transform
    pts1 = left_top_width_height(259, 191, 234, 299)
    pts2 = left_top_width_height(0, 0, 200, 200)

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(im, matrix, (200, 200))
    # Wrap the transformed image

    cv2.imshow('frame', im) # Inital Capture
    cv2.imshow('frame1', result) # Transformed Capture

    if cv2.waitKey(24) == 27:
        break

cv2.destroyAllWindows()