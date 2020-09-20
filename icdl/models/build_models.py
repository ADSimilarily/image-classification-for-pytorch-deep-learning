from .build_backone import build_backbone
from .metric_net import MetricNet


def build_models(cfg):
    if cfg.MODEL.TYPE.lower() == 'cls':
        return build_backbone(cfg)
    elif cfg.MODEL.TYPE.lower() == 'metric':
        return MetricNet(cfg)
    else:
        print("不支持的类型，请检查cfg.MODEL.TYPE关键字")