import torch
import torch.nn as nn
from yacs.config import CfgNode

from icdl.utils import comm
from .build_backone import build_backbone
from .backbone.utils import load_weight_without_strict
from .layers.pooling import SelectAdaptivePool2d
from .layers.batch_norm import get_norm
from .layers.create_act import get_act_layer


class MetricNet(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(MetricNet, self).__init__()
        self.embedding_num = cfg.MODEL.METRIC.DIM
        self.backbone = build_backbone(cfg)

        if cfg.MODEL.BACKBONE_WEIGHTS:
            pretrained_dict = torch.load(cfg.MODEL.BACKBONE_WEIGHTS, map_location='cpu')
            load_weight_without_strict(self.backbone, pretrained_dict)

        self.pool_layer = SelectAdaptivePool2d(pool_type=cfg.MODEL.METRIC.POOLING_LAYER, flatten=False)

        gpus = comm.get_world_size()
        norm_layer = cfg.MODEL.METRIC.NORM_LAYER
        if gpus > 1 and norm_layer not in ['SyncBN', 'nnSyncBN', 'naiveSyncBN']:
            norm_layer = 'SyncBN'
        classifier = self.backbone.get_classifier()
        self.batch_norm = get_norm(norm_layer)(classifier.in_features)
        self.relu = get_act_layer(cfg.MODEL.ACTIVATE)(inplace=True)
        self.linear = nn.Linear(classifier.in_features, self.embedding_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(inputs)
        features = self.pool_layer(features)
        output = self.batch_norm(features)
        output = self.relu(output)
        output = torch.flatten(output, 1)

        output = self.linear(output)
        denom = output.norm(2, 1, True).clamp_min(1e-12)
        output = output / denom

        return output

    def get_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)
