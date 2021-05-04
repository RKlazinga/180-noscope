import json
import os

import cv2
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent
from PyQt5.QtWidgets import *
import numpy as np


padding = 60
radius = 100
# offset is the distance between the center-line and the corner we are targeting
# on 389 radius the offset is 63, so
offset = radius * 63/389

size = 2 * padding + 2 * radius


class UI(QWidget):
    IM_WIDTH, IM_HEIGHT = 800, 500

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Labeler")

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.im_view = QGraphicsView()
        self.im_scene = QGraphicsScene()
        self.im_view.setScene(self.im_scene)

        self.im = ClickablePixmapItem()
        self.im_scene.addItem(self.im)

        self.im_scale = None
        self.im_path = None
        self.im_cv_raw = None
        self.im_warped = None

        self.perspective_source = [
            [100, 0],
            [100, 200],
            [0, 100],
            [200, 100]
        ]

        self.perspective_target = np.float32([
            [padding + radius - offset, padding],
            [padding + radius - offset, padding + 2*radius],
            [padding, padding + radius - offset],
            [padding + 2*radius, padding + radius - offset]
        ])

        # add perspective markers
        self.marker_size = QPixmap("perspective_shift/marker.png").size()
        for i in range(4):
            self.im_scene.addItem(PerspectiveMarker(i, self.marker_moved_callback, self.perspective_source[i]))
        self.perspective_source = np.float32(self.perspective_source)

        self.im_view.setMinimumSize(self.IM_WIDTH, self.IM_HEIGHT)
        self.layout.addWidget(self.im_view, 0, 0, 2, 1)

        self.sample_output = QLabel()
        self.sample_output.setMinimumSize(size, size)
        self.layout.addWidget(self.sample_output, 0, 1)

        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout, 1, 1)

        self.info_label = QLabel("")
        self.control_layout.addWidget(self.info_label)

        self.next_btn = QPushButton("Save and load next [SPACEBAR]")
        self.control_layout.addWidget(self.next_btn)
        self.next_btn.clicked.connect(self.next_image)

        self.control_layout.addStretch()
        self.next_image()
        self.show()

    def next_image(self):
        # if an image is loaded, save the perspective coords and the output image
        if self.im_path:
            self.im_warped.save(f"corrected_data/{self.im_path}")
            with open(f"corrected_data/{os.path.splitext(self.im_path)[0]}.json", "w") as writefile:
                writefile.write(json.dumps(self.perspective_source.tolist()))

        new_im = self.get_raw_image()
        if new_im:
            self.open_image(new_im)

    def get_raw_image(self):
        raw_images = set(os.listdir("raw_data"))
        corrected_images = set(os.listdir("corrected_data"))

        uncorrected_images = raw_images.difference(corrected_images)
        if len(uncorrected_images) == 0:
            self.info_label.setText("All images labeled!")
            return None
        else:
            self.info_label.setText(f"{len(uncorrected_images)} of {len(raw_images)} remaining")
            return uncorrected_images.pop()

    def open_image(self, path):
        self.im_path = path

        pixmap = QPixmap(f"raw_data/{self.im_path}")
        self.im.setPixmap(pixmap)

        scale = max(pixmap.width() / self.IM_WIDTH, pixmap.height() / self.IM_HEIGHT)
        self.im_scale = scale
        self.im.setScale(1 / scale)

        self.im_cv_raw = cv2.imread(f"raw_data/{self.im_path}")

    def marker_moved_callback(self, marker_idx, pos: QPointF):
        self.perspective_source[marker_idx][0] = (pos.x() + self.marker_size.width()//2) * self.im_scale
        self.perspective_source[marker_idx][1] = (pos.y() + self.marker_size.height()//2) * self.im_scale

        self.recompute_warped()

    def recompute_warped(self):
        matrix = cv2.getPerspectiveTransform(self.perspective_source,
                                             self.perspective_target)

        result = cv2.warpPerspective(self.im_cv_raw, matrix, (size, size))
        height, width, channel = result.shape
        self.im_warped = QImage(result.data, width, height, 3 * width, QImage.Format_BGR888)
        self.sample_output.setPixmap(QPixmap(self.im_warped))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.next_image()


class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        pass


class PerspectiveMarker(QGraphicsPixmapItem):
    def __init__(self, idx, callback, pos):
        super().__init__()
        self.idx = idx
        self.callback = callback

        self.setPos(*pos)
        self.setPixmap(QPixmap("perspective_shift/marker.png"))

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        self.callback(self.idx, self.pos())
        self.setPos(self.pos() + (event.pos() - event.lastPos()))
