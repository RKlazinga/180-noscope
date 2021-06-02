from PyQt5.QtGui import QPixmap, QColor, QKeyEvent
from PyQt5.QtWidgets import *


class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, filename, scale, callback):
        super().__init__()
        self.idx = int(filename.split(".")[0])
        self.callback = callback
        self.setPixmap(QPixmap(f"perspective_shift/assets/{filename}"))
        self.count = 0
        self.setScale(scale)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == 1:
            self.set_count(min(3, self.count + 1))
        elif event.button() == 2:
            self.set_count(max(0, self.count - 1))

    def set_count(self, count):
        self.count = count
        self.callback(self.idx, self.count)
        color_tup = [
            [0, 0, 255],
            [0, 255, 255],
            [255, 0, 255],
        ][min(count-1, 2)]
        color = QColor(*color_tup)

        self.effect = QGraphicsColorizeEffect()
        self.effect.setColor(color)
        self.effect.setStrength(0.8)

        if self.count == 0:
            self.effect.setEnabled(False)
        self.setGraphicsEffect(self.effect)


class PerspectiveMarker(QGraphicsPixmapItem):
    def __init__(self, idx, click_callback, move_callback, pos):
        super().__init__()
        self.idx = idx
        self.click_callback = click_callback
        self.move_callback = move_callback

        self.setPos(*pos)
        self.setPixmap(QPixmap(f"perspective_shift/marker{idx}.png"))

    def mousePressEvent(self, event):
        self.click_callback(self)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        self.move_callback(self.idx, self.pos())
        self.setPos(self.pos() + (event.pos() - event.lastPos()))
