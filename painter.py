import numpy as np
from PyQt5 import Qt, QtCore


class Painter(Qt.QWidget):

    def __init__(self):
        super().__init__()
        self.point = Qt.QPoint(0, 0)
        self.point_center = Qt.QPoint(0, 0)
        self.img = None
        self.scaled_img = None
        self.initial_size = None

        self.left_click = True
        self._endPos = Qt.QPoint(0, 0)
        self._startPos = Qt.QPoint(0, 0)

        self.initial_flag = True

    def load_image(self, img):
        height, width, channel = img.shape
        q_image = Qt.QImage(img.data, width, height, 3 * width, Qt.QImage.Format_RGB888).rgbSwapped()
        self.img = Qt.QPixmap.fromImage(q_image)

    def adaptive_resize(self):
        if self.img is None:
            dummy_img = np.ones((5, 5, 3)).astype(np.uint8) * 128
            height, width, channel = dummy_img.shape
            q_image = Qt.QImage(dummy_img.data, width, height, 3 * width, Qt.QImage.Format_RGB888).rgbSwapped()
            self.img = Qt.QPixmap.fromImage(q_image)
        image_width = self.img.width()
        image_height = self.img.height()
        image_w_h_ratio = image_width / image_height
        parent_w_h_ratio = self.width() / self.height()
        if image_w_h_ratio > parent_w_h_ratio:
            # 横向撑满
            target_width = self.width()
            target_height = image_height * target_width / image_width
            self.setGeometry(0, 0, self.width(), self.height())
            self.point_center = Qt.QPoint(0, (self.height() - target_height) // 2)
            self.point = self.point_center
        else:
            # 纵向撑满
            target_height = self.height()
            target_width = image_width * target_height / image_height
            self.setGeometry(0, 0, self.width(), self.height())
            self.point_center = Qt.QPoint((self.width() - target_width) // 2, 0)
            self.point = self.point_center

        self.initial_size = Qt.QSize(target_width, target_height)
        self.scaled_img = self.img.scaled(Qt.QSize(target_width, target_height))

    def paintEvent(self, e):
        if self.initial_flag:
            self.adaptive_resize()

        painter = Qt.QPainter()
        painter.begin(self)
        painter.drawPixmap(self.point, self.scaled_img)
        painter.end()

    def mouseMoveEvent(self, e):
        # 重写移动事件
        if self.left_click:
            self._endPos = e.pos() - self._startPos
            self.point = self.point + self._endPos
            self._startPos = e.pos()
            self.initial_flag = False
            self.repaint()
            self.initial_flag = True

    def mousePressEvent(self, e):
        if e.button() == Qt.Qt.LeftButton:
            # 点击鼠标左键
            self.left_click = True
            self._startPos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.Qt.LeftButton:
            # 释放鼠标左键
            self.left_click = False
        elif e.button() == Qt.Qt.RightButton:
            # 释放鼠标右键
            self.point = self.point_center
            self.scaled_img = self.img.scaled(self.initial_size)
            self.initial_flag = False
            self.repaint()
            self.initial_flag = True

    def wheelEvent(self, e):
        if e.angleDelta().y() < 0:
            # 放大图片
            self.scaled_img = self.img.scaled(int(self.scaled_img.width() * 0.8), int(self.scaled_img.height()*0.8))
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() + 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() + 5)
            self.point = Qt.QPoint(new_w, new_h)
            self.initial_flag = False
            self.repaint()
            self.initial_flag = True
        elif e.angleDelta().y() > 0:
            # 缩小图片
            self.scaled_img = self.img.scaled(int(self.scaled_img.width()*1.2), int(self.scaled_img.height()*1.2))
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() - 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() - 5)
            self.point = Qt.QPoint(new_w, new_h)
            self.initial_flag = False
            self.repaint()
            self.initial_flag = True

    # def resizeEvent(self, e):
    #     if self.parent is not None:
    #         self.scaled_img = self.img.scaled(self.initial_size)
    #         self.point = self.point_center
    #         self.update()
