from .launch import launch
from .registry import ENFINE_REGISTRY
from .trainers import *
from .testers import *


def train_net(cfg):
    if cfg.MODEL.TYPE == 'cls':
        runner = ENFINE_REGISTRY.get('CLSTrainer')(cfg)
    else:
        runner = ENFINE_REGISTRY.get('METRICTrainer')(cfg)
    runner()


def test_net(cfg):
    if cfg.TEST.NAME:
        runner = ENFINE_REGISTRY.get(cfg.TEST.NAME)(cfg)
    else:
        runner = ENFINE_REGISTRY.get(cfg.MODEL.TYPE.upper() + 'Tester')(cfg)
    runner()


def build_runner(cfg, opt):
    if opt == 'train':
        launch(train_net, cfg)
    elif opt == 'infer':
        test_net(cfg)
    elif opt == 'tmp':
        ENFINE_REGISTRY.get(cfg.TEST.METRIC.TEMPLATE_NAME)(cfg)
    elif opt == 'jit' or opt == 'onnx':
        ENFINE_REGISTRY.get(opt)(cfg)
    else:
        print('无效操作，请检查opt参数')