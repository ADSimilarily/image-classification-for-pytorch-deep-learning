from .registry import TRANSFORMS_REGISTRY
import albumentations as AL
from .transform_actions import Filling


class BaseTransform:
    def __init__(self):
        self.transforms = None

    def __call__(self, **data):
        return self.transforms(**data)


@TRANSFORMS_REGISTRY.register()
class DefaultTransform(BaseTransform):
    def __init__(self, width, height):
        super().__init__()
        max_size = int(max(width, height) * 1.05)

        self.transforms = AL.Compose([
            AL.LongestMaxSize(max_size),
            AL.IAAPerspective(scale=(0.02, 0.04), keep_size=True, p=0.3),
            AL.OneOf([AL.HorizontalFlip(p=1), AL.VerticalFlip(p=1), AL.Transpose(p=1)], p=0.7),
            AL.RandomRotate90(p=0.5),
            AL.OneOf([
                AL.RandomBrightnessContrast(p=0.8),
                AL.RandomGamma(p=0.8),
                AL.HueSaturationValue(p=0.3)]),
            Filling(max_size),
            AL.CoarseDropout(
                max_holes=12,
                max_height=24,
                max_width=24,
                min_holes=4,
                min_height=8,
                min_width=8,
                p=0.3),
            AL.OneOf([AL.Resize(height, width), AL.RandomCrop(height, width)], p=1),
        ])


@TRANSFORMS_REGISTRY.register()
class InferTransform(BaseTransform):
    def __init__(self, width, height):
        super().__init__()
        max_size = max(width, height)

        self.transforms = AL.Compose([
            AL.LongestMaxSize(max_size),
            Filling(max_size),
            AL.Resize(height, width, p=1)
        ])