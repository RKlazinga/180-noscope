import json
import os
import cv2
import numpy as np

from PyQt5.QtCore import QPointF, Qt, QLine, QLineF
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QColor, QPen
from PyQt5.QtWidgets import *

from data_collection.perspective_shift.label_ui.items import PerspectiveMarker, ClickablePixmapItem


padding = 60
radius = 100
# offset is the distance between the center-line and the corner we are targeting
# on 389 radius the offset is 63, so
offset = radius * 63/389

size = 2 * padding + 2 * radius


class UI(QWidget):
    IM_WIDTH, IM_HEIGHT = 1100, size*2

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Labeler")

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.im_view = QGraphicsView()
        self.im_scene = QGraphicsScene()
        self.im_view.setScene(self.im_scene)

        self.im = QGraphicsPixmapItem()
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
            [padding + radius + offset, padding],
            [padding + radius - offset, padding + 2*radius],
            [padding, padding + radius - offset],
            [padding + 2*radius, padding + radius + offset]
        ])

        # add connecting lines between markers
        self.vertical_line: QGraphicsLineItem = self.im_scene.addLine(125, 25, 125, 225)
        self.vertical_line.setPen(QColor(255, 255, 255, 100))
        self.horizontal_line: QGraphicsLineItem = self.im_scene.addLine(25, 125, 225, 125)
        self.horizontal_line.setPen(QColor(255, 255, 255, 100))

        # add perspective markers
        self.marker_size = QPixmap("data_collection/perspective_shift/marker.png").size()
        markers = []
        for i in range(4):
            marker = PerspectiveMarker(i,
                                       self.marker_clicked_callback,
                                       self.marker_moved_callback,
                                       self.perspective_source[i])
            self.im_scene.addItem(marker)
        self.highlighted_marker: PerspectiveMarker = None

        self.perspective_source = np.float32(self.perspective_source)

        self.im_view.setFixedSize(self.IM_WIDTH, self.IM_HEIGHT)
        self.im_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.im_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout.addWidget(self.im_view, 0, 0, 3, 1)

        self.sample_output = QLabel()
        self.sample_output.setMinimumSize(size, size)
        self.layout.addWidget(self.sample_output, 0, 1)

        self.classification_box = QGraphicsView()
        self.classification_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.classification_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.classification_box.setFixedSize(size, size)
        self.classification_box_scene = QGraphicsScene()
        self.classification_box.setScene(self.classification_box_scene)

        bg_item = QGraphicsPixmapItem(QPixmap("data_collection/perspective_shift/assets/bg.png"))
        bg_item.setScale(size/460)
        self.classification_box_scene.addItem(bg_item)

        self.count_dict = dict()
        for i in os.listdir("data_collection/perspective_shift/assets"):
            if i != "bg.png":
                item = ClickablePixmapItem(i, size/460, self.classification_clicked_callback)
                self.classification_box_scene.addItem(item)
                item.set_count(0)

        self.layout.addWidget(self.classification_box, 1, 1)

        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout, 2, 1)

        self.info_label = QLabel("")
        self.control_layout.addWidget(self.info_label)

        self.next_btn = QPushButton("Save and load next [SPACEBAR]")
        self.control_layout.addWidget(self.next_btn)
        self.next_btn.clicked.connect(self.next_image)

        self.skip_btn = QPushButton("Skip")
        self.control_layout.addWidget(self.skip_btn)
        self.skip_btn.clicked.connect(self.skip_image)

        self.control_layout.addStretch()
        self.next_image()
        self.show()

        for m in markers:
            self.marker_moved_callback(m)

    def classification_clicked_callback(self, idx, new_count):
        self.count_dict[idx] = new_count

    def next_image(self):
        # if an image is loaded, save the perspective coords and the output image
        if self.im_path:
            self.im_warped.save(f"data/corrected/{self.im_path}")

            # save coords and dart count
            data = {
                "perspective": self.perspective_source.tolist(),
                "darts": self.count_dict
            }
            with open(f"data/corrected/{os.path.splitext(self.im_path)[0]}.json", "w") as writefile:
                writefile.write(json.dumps(data))

            # reset dart count
            for i in self.classification_box_scene.items():
                if isinstance(i, ClickablePixmapItem):
                    i.set_count(0)

        new_im = self.get_raw_image()
        if new_im:
            self.open_image(new_im)

    def skip_image(self):
        file_exists = os.path.isfile("data/corrected/_skip.txt")

        with open("data/corrected/_skip.txt", "a") as writefile:
            if file_exists:
                writefile.write("|")
            writefile.write(self.im_path)

        self.im_path = None
        self.next_image()

    def get_raw_image(self):
        raw_images = set(os.listdir("data/raw"))
        corrected_images = set(os.listdir("data/corrected"))

        uncorrected_images = raw_images.difference(corrected_images)

        if os.path.isfile("data/corrected/_skip.txt"):
            with open("data/corrected/_skip.txt", "r") as readfile:
                skipped_images = set(readfile.read(-1).split("|"))
                uncorrected_images = uncorrected_images.difference(skipped_images)

        if len(uncorrected_images) == 0:
            self.info_label.setText("All images labeled!")
            self.im_path = None
            return None
        else:
            self.info_label.setText(f"{len(uncorrected_images)} of {len(raw_images)} remaining")
            return uncorrected_images.pop()

    def open_image(self, path):
        self.im_path = path

        pixmap = QPixmap(f"data/raw/{self.im_path}")
        self.im.setPixmap(pixmap)

        scale = max(pixmap.width() / self.IM_WIDTH, pixmap.height() / self.IM_HEIGHT)
        self.im_scale = scale
        self.im.setScale(1 / scale)

        self.im_cv_raw = cv2.imread(f"data/raw/{self.im_path}")

    def marker_clicked_callback(self, marker: PerspectiveMarker):
        if self.highlighted_marker:
            self.highlighted_marker.setGraphicsEffect(None)

        self.highlighted_marker = marker
        halo = QGraphicsColorizeEffect()
        halo.setColor(QColor(240, 0, 100))
        halo.setStrength(0.8)
        marker.setGraphicsEffect(halo)

    def marker_moved_callback(self, marker: PerspectiveMarker):
        self.perspective_source[marker.idx][0] = (marker.pos().x() + self.marker_size.width()//2) * self.im_scale
        self.perspective_source[marker.idx][1] = (marker.pos().y() + self.marker_size.height()//2) * self.im_scale

        if marker.idx < 2:
            line = self.vertical_line
        else:
            line = self.horizontal_line

        delta = QPointF(self.marker_size.width() / 2, self.marker_size.height() / 2)
        if marker.idx % 2 == 1:
            a, b = marker.pos() + delta, line.line().p2()
        else:
            a, b = line.line().p1(), marker.pos() + delta

        line.setLine(QLineF(a, b))

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
        if self.highlighted_marker:
            if event.key() == Qt.Key_W:
                self.highlighted_marker.moveBy(0, -1)
            if event.key() == Qt.Key_A:
                self.highlighted_marker.moveBy(-1, 0)
            if event.key() == Qt.Key_S:
                self.highlighted_marker.moveBy(0, 1)
            if event.key() == Qt.Key_D:
                self.highlighted_marker.moveBy(1, 0)
            self.marker_moved_callback(self.highlighted_marker)
