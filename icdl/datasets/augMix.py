from .registry import TRANSFORMS_REGISTRY
from .transform_actions import Filling
from .transform import BaseTransform
import albumentations as AL
import numpy as np
import cv2


@TRANSFORMS_REGISTRY.register()
class ResizeTransform(BaseTransform):
    def __init__(self, width, height):
        super().__init__()
        scale = int(max(width, height) * 1.05)

        self.transforms = AL.Compose([
            AL.LongestMaxSize(scale),
            Filling(scale),
            AL.OneOf([AL.Resize(width, height), AL.RandomCrop(width, height)], p=1),
        ])


@TRANSFORMS_REGISTRY.register()
class DefaultAugMix(BaseTransform):
    def __init__(self):
        super().__init__()

        self.transforms = AL.Compose([
            AL.IAAPerspective(scale=(0.02, 0.04), keep_size=True, p=0.3),
            AL.RandomBrightnessContrast(p=0.3),
            AL.RandomGamma(p=0.3),
            AL.GaussianBlur(blur_limit=9, p=0.4),
            AL.GaussNoise(p=0.3),
            AL.RandomRotate90(p=0.6),
            AL.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.2),
            AL.CoarseDropout(max_holes=24,
                            max_height=24,
                            max_width=24,
                            min_holes=8,
                            min_height=8,
                            min_width=8,
                            p=0.4),
        ])


@TRANSFORMS_REGISTRY.register()
class AugMixAugment:
    """ AugMix Transform
    Adapted and improved from impl here: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """
    def __init__(self, ops, resize_trans, alpha=1., width=3, depth=-1, blended=False):
        self.ops = ops.transforms.transforms.transforms
        self.alpha = alpha
        self.width = width
        self.depth = depth
        self.blended = blended  # blended mode is faster but not well tested
        self.resize_trans = resize_trans

    def _calc_blended_weights(self, ws, m):
        ws = ws * m
        cump = 1.
        rws = []
        for w in ws[::-1]:
            alpha = w / cump
            cump *= (1 - alpha)
            rws.append(alpha)
        return np.array(rws[::-1], dtype=np.float32)

    def _apply_blended(self, img, mixing_weights, m):
        # This is my first crack and implementing a slightly faster mixed augmentation. Instead
        # of accumulating the mix for each chain in a Numpy array and then blending with original,
        # it recomputes the blending coefficients and applies one PIL image blend per chain.
        # TODO the results appear in the right ballpark but they differ by more than rounding.
        img_orig = img.copy()
        ws = self._calc_blended_weights(mixing_weights, m)
        for w in ws:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img_orig  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(image=img_aug)["image"]

            if img.shape != img_aug.shape:
                img = np.transpose(img, (1, 0, 2))
            img = cv2.addWeighted(img, w, img_aug, 1 - w, 0.0)
        return img

    def _apply_basic(self, img, mixing_weights, m):
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        mixed = np.zeros(img.shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(image=img_aug)["image"]
            if mixed.shape != img_aug.shape:
                mixed = np.transpose(mixed, (1, 0, 2))
            mixed += mw * img_aug.astype(np.float32)
        np.clip(mixed, 0, 255., out=mixed)
        mixed = mixed.astype(np.uint8)

        if mixed.shape != img.shape:
            img = np.transpose(img, (1, 0, 2))

        return cv2.addWeighted(img, m, mixed, 1 - m, 0.0)

    def __call__(self, image):
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        if self.blended:
            mixed = self._apply_blended(image, mixing_weights, m)
        else:
            mixed = self._apply_basic(image, mixing_weights, m)
        return self.resize_trans(image=mixed)