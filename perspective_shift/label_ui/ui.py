import cv2
from PyQt5.QtCore import QPoint, QPointF
from PyQt5.QtGui import QPixmap, QImage
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

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.im_view = QGraphicsView()
        self.im_scene = QGraphicsScene()
        self.im_view.setScene(self.im_scene)

        self.im = ClickablePixmapItem()
        self.im_scene.addItem(self.im)

        self.im_scale = 1

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
        self.marker_size = QPixmap("marker.png").size()
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

        self.next_btn = QPushButton("Next")
        self.control_layout.addWidget(self.next_btn)

        self.control_layout.addStretch()
        self.open_image("01.jpg")
        self.show()

    def open_image(self, path):
        pixmap = QPixmap(path)
        self.im.setPixmap(pixmap)

        scale = max(pixmap.width() / self.IM_WIDTH, pixmap.height() / self.IM_HEIGHT)
        self.im_scale = scale
        self.im.setScale(1 / scale)

    def marker_moved_callback(self, marker_idx, pos: QPointF):
        self.perspective_source[marker_idx][0] = (pos.x() + self.marker_size.width()//2) * self.im_scale
        self.perspective_source[marker_idx][1] = (pos.y() + self.marker_size.height()//2) * self.im_scale

        self.recompute_warped()

    def recompute_warped(self):
        matrix = cv2.getPerspectiveTransform(self.perspective_source,
                                             self.perspective_target)
        im = cv2.imread("01.jpg")
        result = cv2.warpPerspective(im, matrix, (size, size))
        height, width, channel = result.shape
        q_image = QImage(result.data, width, height, 3 * width, QImage.Format_BGR888)
        self.sample_output.setPixmap(QPixmap(q_image))


class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        print(event.pos())


class PerspectiveMarker(QGraphicsPixmapItem):
    def __init__(self, idx, callback, pos):
        super().__init__()
        self.idx = idx
        self.callback = callback

        self.setPos(*pos)
        # TODO improve marker PNG
        self.setPixmap(QPixmap("marker.png"))

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        self.callback(self.idx, self.pos())
        self.setPos(self.pos() + (event.pos() - event.lastPos()))


if __name__ == '__main__':
    app = QApplication([])
    ui = UI()
    app.exec_()