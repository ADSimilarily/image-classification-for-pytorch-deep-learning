import random
import numpy as np
from typing import List, Tuple

__all__ = ["Filling", "Jitter"]


def box_intersection(a: List or Tuple, b: List or Tuple) -> Tuple:
    # [left, right, top, bot]
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    top = max(a[2], b[2])
    bot = min(a[3], b[3])
    return (left, right, top, bot)


class Filling(object):
    def __init__(self, size, p=1.0):
        self.size = size
        self.p =p

    def __call__(self, image, force_apply=False, **kwargs):
        if random.random() > self.p:
            return image
        h, w = image.shape[:2]
        self.size = max(self.size, h, w)
        if len(image.shape) == 2:
            out = np.zeros((self.size, self.size), np.uint8)
        elif len(image.shape) == 3:
            out = np.zeros((self.size, self.size, 3), np.uint8)
        else:
            return image
        y1 = (self.size - h) // 2
        y2 = y1 + h
        x1 = (self.size - w) // 2
        x2 = x1 + w

        out[y1:y2, x1:x2] = image
        return {"image":out.copy()}


class Jitter(object):
    def __init__(self, jitter, p=0.5):
        if isinstance(jitter, float):
            assert 0 < jitter <= 1.0
            jitter = (min(-jitter, jitter), max(-jitter, jitter))
        elif isinstance(jitter, (list, tuple)):
            assert 0 < min(jitter) < max(jitter) <= 1.0
            jitter = (min(jitter), max(jitter))

        self.jitter = jitter
        self.p = p

    def _random_scale(self, jitter, orig):
        ratio = random.random() * (jitter[1] - jitter[0]) + jitter[0]
        return int(orig * ratio)

    def __call__(self, image, force_apply=False, **kwargs):
        if random.random() > self.p:
            return image

        shape = list(image.shape)
        h, w = shape[:2]
        pleft = self._random_scale(self.jitter, w)
        pright = w - self._random_scale(self.jitter, w)
        ptop = self._random_scale(self.jitter, h)
        pbot = h - self._random_scale(self.jitter, h)

        swidth = pright - pleft
        sheight = pbot - ptop
        shape[:2] = [sheight, swidth]

        dst = np.zeros(shape, dtype=image.dtype)

        I = box_intersection((pleft, pright, ptop, pbot), (0, w, 0, h))
        dleft = max(0, -pleft)
        dright = dleft + I[1] - I[0]
        dtop = max(0, -ptop)
        dbot = dtop + I[3] - I[2]

        dst[dtop:dbot, dleft:dright] = image[I[2]:I[3], I[0]:I[1]]

        return {"image": dst.copy()}
