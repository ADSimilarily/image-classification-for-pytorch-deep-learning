# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List
import torch
from yacs.config import CfgNode

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR, WarmupPloyLR


def build_optimer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    lr = cfg.SOLVER.BASE_LR
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optim_name = cfg.SOLVER.OPTIMER.lower()
    if optim_name == 'sgd':
        return torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif optim_name == 'adam':
        return torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
    elif optim_name == 'adamw':
        return torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR)
    else:
        raise ValueError("仅支持以下算法：sgd、adam、adamw。请重新设置优化算法")


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer, dataset_size: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    max_iters = cfg.SOLVER.MAX_ITER * dataset_size // cfg.TRAIN.INPUT_BATCH

    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            max_iters,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupPloyLR":
        return WarmupPloyLR(
            optimizer,
            max_iters,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
