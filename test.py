import inspect
from utils import cv2_utils as cv2

from albumentations.augmentations.transforms import __all__ as a_all
from albumentations.imgaug.transforms import __all__ as i_all
from albumentations.pytorch.transforms import __all__ as t_all
import albumentations as A

from utils.ann_utils import BaseSample


def f(a, b, c=10, d=12, *args, **kwargs):
    pass


print(inspect.signature(f))
print(inspect.getfullargspec(f))

print(len(a_all) + len(i_all) + len(t_all))

for class_name in a_all:
    class_ = getattr(A, class_name)
    print(class_name, ':', inspect.signature(class_))



trs = A.CenterCrop(500, 500)
img_path = r'E:\InspectionProjects\latest_pg_bottle\dataset\JPEGImages\0630142746_4.jpg'
ann_path = r'E:\InspectionProjects\latest_pg_bottle\dataset\Annotations\0630142746_4.json'
sample = BaseSample()
sample.load(ann_path)
image = cv2.imread(img_path)
h, w, c = image.shape
bboxes = [ann['bbox'] for ann in sample.annotations]
bboxes = [(b[0]/w, b[1]/h, (b[0]+b[2])/w, (b[1]+b[3])/h) for b in bboxes]
print(bboxes)
transformed = trs(image=image, bboxes=bboxes)
cv2.imshow('show', transformed['image'])
cv2.waitKey()

print(transformed['bboxes'])
