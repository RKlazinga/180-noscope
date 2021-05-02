import cv2
import numpy as np

while True:
    im = cv2.imread("test.jpg")

    padding = 60
    radius = 100
    # offset is the distance between the center-line and the corner we are targeting
    # on 389 radius the offset is 63, so
    offset = radius * 63/389

    size = 2 * padding + 2 * radius

    # Locate points of the documents or object which you want to transform
    pts1 = np.float32([
        [369, 227],  # top
        [381, 457],  # bottom
        [292, 331],  # left
        [469, 319]   # right
    ])
    pts2 = np.float32([
        [padding + radius - offset, padding],
        [padding + radius - offset, padding + 2*radius],
        [padding, padding + radius - offset],
        [padding + 2*radius, padding + radius - offset]
    ])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(im, matrix, (size, size))
    # Wrap the transformed image

    cv2.imshow('frame', im)  # Inital Capture
    cv2.imshow('frame1', result)  # Transformed Capture

    if cv2.waitKey(24) == 27:
        break

cv2.destroyAllWindows()
