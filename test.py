import inspect

from albumentations.augmentations.transforms import __all__ as a_all
from albumentations.imgaug.transforms import __all__ as i_all
from albumentations.pytorch.transforms import __all__ as t_all
import albumentations as A


def f(a, b, c=10, d=12, *args, **kwargs):
    pass

print(inspect.signature(f))
print(inspect.getfullargspec(f))

print(len(a_all) + len(i_all) + len(t_all))

for class_name in __all__:
    class_ = getattr(A, class_name)
    print(class_name, ':', inspect.signature(class_))
