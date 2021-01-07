import json
import math
import numpy as np
# from utils.ann_utils import get_bboxes_and_polygons_from_mask, get_bbox_from_polygon, get_palette
from utils import cv2_utils as cv2
from utils import path_utils as path

from PIL import Image
from pycocotools import mask as coco_mask


def get_palette(num_classes=256, count=8):
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = 0
        g = 0
        b = 0
        id_ = i
        for j in range(7):
            str_id = ''.join([str((id_ >> y) & 1) for y in range(count - 1, -1, -1)])
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id_ = id_ >> 3
        color_map[i, 0] = r
        color_map[i, 1] = g
        color_map[i, 2] = b
    return color_map


def get_bbox_from_polygon(polygon):
    x_min = np.min(polygon[0][0::2])
    y_min = np.min(polygon[0][1::2])
    x_max = np.max(polygon[0][0::2])
    y_max = np.max(polygon[0][1::2])
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox


def get_bboxes_and_polygons_from_mask(mask):
    bbox_list = []
    polygon_list = []
    ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contour_list, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def filter_func(c):
        if len(c) <= 2:
            return False
        else:
            return True

    contour_list = list(filter(filter_func, contour_list))

    for contour in contour_list:

        polygon = [[]]
        for point in contour:
            x = float(point[0][0])
            y = float(point[0][1])
            polygon[0].extend([x, y])
        bbox = get_bbox_from_polygon(polygon)
        polygon_list.append(polygon)
        bbox_list.append(bbox)

    return bbox_list, polygon_list


class GeneralAnnotation:
    def __init__(self, labels):
        self.image_name = ''
        self.image_height = 0
        self.image_width = 0
        self.annotations = []
        self.labels = labels
        self.palette = get_palette()
        self.colors = self.palette.tolist()

    def load(self, json_path):
        raise NotImplementedError

    def viz(self, image_path, modes='L-B-P-M'):
        """

        :param image_path:
        :param modes: L - Label
                      B - BBox
                      P - Poly
                      M - Mask
        :param labels:
        :return:
        """
        modes = modes.split('-')
        show_mask = True if 'M' in modes else False
        show_poly = True if 'P' in modes else False
        show_bbox = True if 'B' in modes else False
        show_label = True if 'L' in modes else False

        img = cv2.imread(image_path)
        raw_img = img.copy()
        img_height, img_width, _ = img.shape
        for ii, ann in enumerate(self.annotations):
            label = ann['label']
            color = tuple(self.colors[self.labels.index(label)])
            mask_color = color
            poly_color = color
            bbox_color = color
            label_color = (255, 255, 255)

            polygon = ann['polygon']

            if show_mask:
                rles = coco_mask.frPyObjects(polygon, img_height, img_width)
                binary = coco_mask.decode(rles)
                binary = binary[:, :, 0]

                mask = np.array([(binary * v).astype(np.uint8) for v in mask_color]).transpose((1, 2, 0))
                # mask 覆盖到原图上
                condition = np.all(mask[:, :] == (0, 0, 0), axis=2)
                condition = np.array([condition, condition, condition]).transpose((1, 2, 0))
                img = np.where(condition, img, mask)
            if show_poly:
                points = [[i, j] for i, j in zip(polygon[0][0::2], polygon[0][1::2])]
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points], True, poly_color, thickness=2)
            if show_bbox:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, thickness=4)
            if show_label:
                num_points = len(polygon[0]) // 2
                x = int(sum(polygon[0][0::2])) // num_points
                y = int(sum(polygon[0][1::2])) // num_points
                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, label_color, thickness=2)

        alpha = 0.3
        img = cv2.addWeighted(img, alpha, raw_img, 1 - alpha, 0)

        return img

    def dump(self, dump_path):
        with open(dump_path, 'w') as f:
            dump_dict = self.__dict__.copy()
            del dump_dict['labels']
            del dump_dict['colors']
            del dump_dict['palette']
            json.dump(dump_dict, f)

    def save_voc_mask(self, save_path):
        polygons = []
        label_ids = []
        for ann in self.annotations:
            polygon = ann['polygon']
            label_id = self.labels.index(ann['label'])
            polygons.append(polygon)
            label_ids.append(label_id)
        full_label_map = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for polygon, label_id in zip(polygons, label_ids):
            if polygon[0]:
                rles = coco_mask.frPyObjects(polygon, self.image_height, self.image_width)
                binary = coco_mask.decode(rles)
                binary = binary[:, :, 0]

                label_map = binary * label_id
                label_map = label_map.astype(np.uint8)

                full_label_map *= (label_map == 0)
                full_label_map += label_map

        mask = Image.fromarray(full_label_map, 'P')
        mask.putpalette(self.palette)
        mask.save(save_path)

    def save_yolo_txt(self, save_path):
        pass


class LabelmeAnnotation(GeneralAnnotation):
    def __init__(self, labels):
        super().__init__(labels)

    def load(self, json_path):
        with open(json_path, 'r') as f:
            json_obj = json.load(f)
        self.image_name = json_obj['imagePath']
        self.image_height = json_obj['imageHeight']
        self.image_width = json_obj['imageWidth']
        for shape in json_obj['shapes']:
            label = shape['label']
            if shape['shape_type'] == 'polygon':
                polygon = [list(np.array(shape['points']).reshape(-1))]
            else:
                polygon = [[]]
                try:
                    x0 = shape['points'][0][0]
                    y0 = shape['points'][0][1]
                    x1 = shape['points'][1][0]
                    y1 = shape['points'][1][1]
                    if shape['shape_type'] == 'rectangle':
                        polygon = [[x0, y0, x1, y0, x1, y1, x0, y1]]
                    if shape['shape_type'] == 'circle':
                        r = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                        for i in range(360 // 15):
                            theta = i * math.pi * 15 / 180
                            x = x0 + r * math.cos(theta)
                            y = y0 + r * math.sin(theta)
                            polygon[0].extend([x, y])
                except IndexError as e:
                    print(e, shape)
            if polygon[0]:
                bbox = get_bbox_from_polygon(polygon)
                self.annotations.append({
                    'label': label,
                    'bbox': bbox,
                    'polygon': polygon
                })
        if self.annotations:
            return True
        else:
            return False


class ColabelerAnnotation(GeneralAnnotation):
    def __init__(self, labels):
        super().__init__(labels)

    def load(self, json_path):
        with open(json_path, 'r') as f:
            json_obj = json.load(f)
        if not (json_obj['labeled'] and 'attachments' in json_obj):
            return False
        self.image_name = path.get_name(json_obj['path'])
        self.image_height = json_obj['size']['height']
        self.image_width = json_obj['size']['width']

        label_name_list = [item['name'] for item in json_obj['outputs']['object']]
        label_img_path_list = sorted([
            path.join(path.get_parent(json_path), 'attachments\\' + path.get_name(item['path']))
            for item in json_obj['attachments']
        ])

        for label_name, label_img_path in zip(label_name_list, label_img_path_list):
            label_img = cv2.imread(label_img_path)
            mask = np.all(label_img != [0, 0, 0], axis=2).astype(np.uint8) * 128
            bbox_list, polygon_list = get_bboxes_and_polygons_from_mask(mask)
            for bbox, polygon in zip(bbox_list, polygon_list):
                self.annotations.append({
                    'label': label_name,
                    'bbox': bbox,
                    'polygon': polygon
                })
        return True


if __name__ == '__main__':
    ann = ColabelerAnnotation(labels=['__background__', 'dirt', 'glue'])
    ann.load(r'E:\InspectionData\__demo_data\colabeler_data\outputs\0630142746_4.json')
    ann.dump(r'tmp\colabeler.json')
    viz_img = ann.viz(r'E:\InspectionData\__demo_data\colabeler_data\0630142746_4.jpg', modes='L-P')
    ann.save_voc_mask(r'tmp\colabeler_mask.png')
    cv2.imshow('show', viz_img)
    cv2.waitKey()

    ann = LabelmeAnnotation(labels=['__background__', '01', 'box'])
    ann.load(r'E:\InspectionData\__demo_data\labelme_data\01-1_16.07.27.json')
    ann.dump(r'tmp\labelme.json')
    viz_img = ann.viz(r'E:\InspectionData\__demo_data\labelme_data\01-1_16.07.27.jpg', modes='L-P')
    ann.save_voc_mask(r'tmp\labelme_mask.png')
    cv2.imshow('show', viz_img)
    cv2.waitKey()
