import cv2
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


class CocoAnnJsonWriter:
    def __init__(self):
        self.label_list = []
        self.info = {}
        self.licenses = []
        self.category_list = []
        self.image_list = []
        self.annotation_list = []

    def add_category(self, cat_id, label):
        self.label_list.append(label)
        self.category_list.append({
            'supercategory': 'object',
            'id': cat_id,
            'name': label
        })

    def load_labels_txt(self, labels_txt_path):
        with open(labels_txt_path, 'r') as f:
            for ii, line in enumerate(f.readlines()[1:]):
                label = line.strip()
                self.add_category(ii+1, label)

    def load_label_list(self, label_list):
        for ii, label in enumerate(label_list):
            self.add_category(ii+1, label)

    def add_image(self, _id, file_name, height, width):
        self.image_list.append({
            'id': _id,
            'file_name': file_name,
            'height': height,
            'width': width
        })

    def add_annotation(self, label, image_id, bbox, segmentation, area):
        self.annotation_list.append({
            'segmentation': segmentation,
            'iscrowd': 0,

            'category_id': self.label_list.index(label) + 1,
            'image_id': image_id,
            'id': len(self.annotation_list),
            'bbox': bbox,
            'area': area
        })

    def save_ann_json(self, path):
        annotation_content = {
            'categories': self.category_list,
            'images': self.image_list,
            'annotations': self.annotation_list
        }
        with open(path, 'w') as f:
            json.dump(annotation_content, f)





def get_mask_from_polygons(polygon_list, label_id_list, height, width):
    full_label_map = np.zeros((height, width), dtype=np.uint8)
    for polygon, label_id in zip(polygon_list, label_id_list):
        if polygon[0]:
            rles = coco_mask.frPyObjects(polygon, height, width)
            binary = coco_mask.decode(rles)
            binary = binary[:, :, 0]

            label_map = binary * label_id
            label_map = label_map.astype(np.uint8)

            full_label_map *= (label_map == 0)
            full_label_map += label_map

    mask = Image.fromarray(full_label_map, 'P')
    mask.putpalette(get_palette())
    return mask



