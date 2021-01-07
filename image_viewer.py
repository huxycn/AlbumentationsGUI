from PyQt5 import Qt
from PyQt5.uic import loadUi
from itertools import compress

from utils.ann_utils import LabelmeAnnotation, ColabelerAnnotation
from painter import Painter


# import sys
# sys.path.append(r'D:\code_Pro\Seizet_DL_Inspection')


class ImageViewer(Qt.QWidget):
    current_idx_changed = Qt.pyqtSignal()
    add_train_state_changed = Qt.pyqtSignal(int, bool)

    def __init__(self, img_path='', ann_path='', parent=None):
        super().__init__(parent)
        loadUi('ImageViewer.ui', self)
        self.img_path = img_path
        self.ann_path = ann_path

        # 显示标注类型开关，初始值
        self.show_total = False
        self.show_label = True
        self.show_bbox = True
        self.show_poly = False
        self.show_mask = False
        
        self.chk_total.setChecked(self.show_total)
        self.set_chk_checked(self.show_label, self.show_bbox, self.show_poly, self.show_mask)
        
        self.chk_total.stateChanged.connect(self.show_total_or_not)
        self.chk_label.stateChanged.connect(self.paint_img)
        self.chk_bbox.stateChanged.connect(self.paint_img)
        self.chk_poly.stateChanged.connect(self.paint_img)
        self.chk_mask.stateChanged.connect(self.paint_img)

        self.painter = Painter()
        layout = Qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.painter)
        self.painter_area.setLayout(layout)

    def set_path(self, img_path, ann_path):
        self.img_path = img_path
        self.ann_path = ann_path
        self.paint_img()
        
    def set_chk_checked(self, show_label, show_bbox, show_poly, show_mask):
        # 是否显示标签
        self.show_label = show_label
        self.chk_label.setChecked(self.show_label)

        # 是否显示矩形
        self.show_bbox = show_bbox
        self.chk_bbox.setChecked(self.show_bbox)

        # 是否显示多边形
        self.show_poly = show_poly
        self.chk_poly.setChecked(self.show_poly)

        # 是否显示蒙版
        self.show_mask = show_mask
        self.chk_mask.setChecked(self.show_mask)

    def show_total_or_not(self):
        self.show_total = self.chk_total.isChecked()
        self.chk_total.setChecked(self.show_total)
        if self.chk_total:
            self.set_chk_checked(True, True, True, True)
        else:
            self.set_chk_checked(False, False, False, False)

    def paint_img(self):
        self.show_label = self.chk_label.isChecked()
        self.show_bbox = self.chk_bbox.isChecked()
        self.show_poly = self.chk_poly.isChecked()
        self.show_mask = self.chk_mask.isChecked()
        mode_list = ['L', 'B', 'P', 'M']
        flag_list = [self.show_label, self.show_bbox, self.show_poly, self.show_mask]
        modes = compress(mode_list, flag_list)
        modes = '-'.join(modes)

        ann = ColabelerAnnotation(labels=['__bg__', 'dirt', 'glue'])
        ann.load(self.ann_path)
        img = ann.viz(self.img_path, modes=modes)

        self.painter.load_image(img)
        self.painter.repaint()
