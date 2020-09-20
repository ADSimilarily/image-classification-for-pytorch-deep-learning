from .metric import *
from .cls import *
from .registry import LOSSES_REGISTRY


def build_criterion(cfg):
    assert cfg.LOSS.NAME
    return LOSSES_REGISTRY.get(cfg.LOSS.NAME)()