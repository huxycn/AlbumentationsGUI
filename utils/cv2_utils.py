import cv2
import numpy as np

from utils import path_utils as path

from cv2 import *


def imread(filename):
    cv_img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def imwrite(filename, cv_img):
    suffix = path.get_suffix(filename)
    cv2.imencode(suffix, cv_img)[1].tofile(filename)


def adaptive_resize(img, box_shape):
    h, w, c = img.shape
    img_w_h_ratio = w / h
    box_w_h_ratio = box_shape[1] / box_shape[0]
    if img_w_h_ratio > box_w_h_ratio:
        # 横向撑满
        target_w = box_shape[1]
        target_h = h * target_w / w
    else:
        # 纵向撑满
        target_h = box_shape[0]
        target_w = w * target_h / h
    img = cv2.resize(img, (int(target_w), int(target_h)))
    return img


def resize_to_height(img, target_h):
    h, w, c = img.shape
    target_w = target_h * w / h
    img = cv2.resize(img, (int(target_w), int(target_h)))
    return img


def resize_to_width(img, target_w):
    h, w, c = img.shape
    target_h = target_w * h / w
    img = cv2.resize(img, (int(target_w), int(target_h)))
    return img
