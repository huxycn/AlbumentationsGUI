from utils import cv2_utils as cv2
from PyQt5 import Qt


BASE_SIZE = 150
LAYOUT = 'V'
# LAYOUT = 'H'


class ClickableQLabelImage(Qt.QLabel):
    clicked = Qt.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, e):
        self.clicked.emit()


class ThumbImageItem(Qt.QWidget):
    img_clicked = Qt.pyqtSignal(int)
    add_train_state_changed = Qt.pyqtSignal(int, bool)

    def __init__(self, parent, thumbnail, idx):
        super().__init__(parent)
        # thumbnail = cv2.adaptive_resize(thumbnail, (size, size))
        if LAYOUT == 'V':
            thumbnail = cv2.resize_to_width(thumbnail, BASE_SIZE)
        if LAYOUT == 'H':
            thumbnail = cv2.resize_to_height(thumbnail, BASE_SIZE)

        # cv2.imshow('show', thumbnail)
        # cv2.waitKey()
        h, w, _ = thumbnail.shape
        self.setFixedSize(w, h)

        # x = size//2 - w // 2
        # y = size//2 - h // 2
        x = 0
        y = 0

        # 显示缩略图，可点击
        self.lbl_img = ClickableQLabelImage(self)
        self.lbl_img.setGeometry(x, y, w, h)
        q_image = Qt.QImage(thumbnail.data, w, h, 3 * w, Qt.QImage.Format_RGB888).rgbSwapped()
        self.lbl_img.setPixmap(Qt.QPixmap.fromImage(q_image))
        self.lbl_img.clicked.connect(lambda: self.img_clicked.emit(idx))

        # 左上角显示图片编号
        self.lbl_idx = Qt.QLabel(str(idx), self)
        self.lbl_idx.setStyleSheet("background: red; color: white")
        self.lbl_idx.setAlignment(Qt.Qt.AlignCenter)
        self.lbl_idx.setGeometry(x, y, 16, 16)

        # 右上角显示是否加入训练单选框
        self.chk_add_train = Qt.QCheckBox('', self)
        self.chk_add_train.setGeometry(x+w-16, y, 16, 16)
        self.chk_add_train.setCheckState(2)
        self.chk_add_train.stateChanged.connect(lambda: self.add_train_state_changed.emit(idx, self.chk_add_train.checkState()))

    def set_img_clicked(self):
        self.lbl_img.setStyleSheet("border: 4px solid red")

    def unset_img_clicked(self):
        self.lbl_img.setStyleSheet("")


class ThumbImageList(Qt.QScrollArea):
    current_idx_changed = Qt.pyqtSignal(int)
    add_train_state_changed = Qt.pyqtSignal(int, bool)

    def __init__(self, img_list, parent=None):
        super().__init__(parent)

        if LAYOUT == 'V':
            dummy_thumb = cv2.resize_to_width(img_list[0], BASE_SIZE)
            self.setFixedWidth(dummy_thumb.shape[1] + 46)
        if LAYOUT == 'H':
            dummy_thumb = cv2.resize_to_height(img_list[0], BASE_SIZE)
            self.setFixedHeight(dummy_thumb.shape[0] + 46)

        self.__thumb_image_item_list = []
        self.__current_idx = 0
        self.__add_train_state_list = []

        scroll_area_widget_contents = Qt.QWidget()
        if LAYOUT == 'V':
            layout = Qt.QVBoxLayout()
        if LAYOUT == 'H':
            layout = Qt.QHBoxLayout()
        layout.setSpacing(10)
        scroll_area_widget_contents.setLayout(layout)

        for idx, img in enumerate(img_list):
            thumb_image_item = ThumbImageItem(self, img, idx)
            self.__thumb_image_item_list.append(thumb_image_item)
            layout.addWidget(thumb_image_item)

            # 默认第一张图片点击
            if idx == self.__current_idx:
                thumb_image_item.set_img_clicked()
            # 默认全部加入训练
            self.__add_train_state_list.append(True)

            # 内部信号发送到外部，控制外部其它控件
            thumb_image_item.img_clicked.connect(self.current_idx_changed.emit)

            # 内部信号处理
            thumb_image_item.img_clicked.connect(self.set_current_idx)
            thumb_image_item.add_train_state_changed.connect(self.add_train_state_changed.emit)

        self.setWidget(scroll_area_widget_contents)

    def set_current_idx(self, idx):
        if idx != self.__current_idx:
            self.__thumb_image_item_list[idx].set_img_clicked()
            self.__thumb_image_item_list[self.__current_idx].unset_img_clicked()
            self.__current_idx = idx

    def get_current_idx(self):
        return self.__current_idx

    def get_add_train_state_list(self):
        return self.__add_train_state_list

    def set_add_train_state(self, idx, state):
        self.__add_train_state_list[idx] = state
        self.__thumb_image_item_list[idx].chk_add_train.setChecked(state)
