import sys
import cv2_utils as cv2
import albumentations as A
import os.path as osp
from PyQt5 import Qt
from PyQt5.uic import loadUi
from painter import Painter
from thumb_image_list import ThumbImageList


class AugToolGUI(Qt.QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('AugToolGUI.ui', self)
        self.image = None
        self.mask = None
        self.aug_image_list = None
        self.aug_mask_list = None

        # raw_img_area
        self.painter = Painter()
        raw_img_area_layout = Qt.QVBoxLayout()
        raw_img_area_layout.setContentsMargins(0, 0, 0, 0)
        raw_img_area_layout.addWidget(self.painter)
        self.raw_img_area.setLayout(raw_img_area_layout)

        # 操作区
        self.btn_open_img.clicked.connect(self.open_img)

        self.btn_start_aug.clicked.connect(self.start_aug)

    def open_img(self):
        img_path, _ = Qt.QFileDialog.getOpenFileName(
            self, '打开图片', 'C:/', "Image files(*.bmp *.jpg *.pbm *.pgm *.png *.ppm *.xbm *.xpm);;All files (*.*)")
        if osp.exists(img_path) and osp.isfile(img_path):
            self.lbl_img_path.setText('图片位置：{}'.format(img_path))
            self.image = cv2.imread(r'E:\InspectionProjects\pg_bottle\dataset\JPEGImages\0630142746_4.jpg')
            self.mask = cv2.imread(r'E:\InspectionProjects\pg_bottle\dataset\Segmentations\0630142746_4.png')
            self.painter.load_image(self.image)
            self.painter.repaint()

    def start_aug(self):
        trs = A.Compose([
            # A.CropNonEmptyMaskIfExists(512, 512),
            # A.CenterCrop(512, 512),
            A.CoarseDropout(min_holes=8, min_height=100, min_width=100, max_height=100, max_width=100, p=1, mask_fill_value=0),
            # A.HorizontalFlip(),
            # A.RandomFog()
        ])

        aug_image_list = []
        aug_mask_list = []
        for i in range(10):
            transformed = trs(image=self.image, mask=self.mask)
            aug_image_list.append(transformed['image'])
            aug_mask_list.append(transformed['mask'])
        self.aug_image_list = ThumbImageList(aug_image_list)
        self.aug_mask_list = ThumbImageList(aug_mask_list)
        aug_img_area_layout = Qt.QHBoxLayout()
        aug_img_area_layout.setContentsMargins(0, 0, 0, 0)
        aug_img_area_layout.addWidget(self.aug_image_list)
        aug_img_area_layout.addWidget(self.aug_mask_list)
        self.aug_img_area.setLayout(aug_img_area_layout)


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)

    w = AugToolGUI()
    w.show()

    sys.exit(app.exec_())