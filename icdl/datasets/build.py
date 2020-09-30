from .transform import *
from .augMix import AugMixAugment
from .registry import TRANSFORMS_REGISTRY
from .folder_dataset import ImageFolder
from .samplers import BalancedIdentitySampler
from ..utils import comm
from torch.utils.data.dataloader import DataLoader as dataloader

import os
import torch


def create_infer_dataloader(path, cfg):
    val_loader = None
    if os.path.exists(path):
        val_transform = TRANSFORMS_REGISTRY.get(cfg.INFERENCE.TRANSFORM)(cfg.TRAIN.INPUT_WIDTH, cfg.TRAIN.INPUT_HEIGHT)
        val_dataset = ImageFolder(path, val_transform)

        val_sampler = None
        if comm.get_world_size() > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        if cfg.MODEL.TYPE == 'cls':
            val_loader = dataloader(val_dataset, batch_size=cfg.INFERENCE.BATCH_SIZE, shuffle=False, sampler=val_sampler, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        else:
            batch_size = cfg.TRAIN.METRIC.NUM_CLASSES * cfg.TRAIN.METRIC.NUM_SAMPLES
            val_loader = dataloader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                    num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
    return val_loader


def build_train_dataloader(cfg):
    train_floder = os.path.join(cfg.TRAIN.DATASETS, "train")

    class_num = len(os.listdir(train_floder))
    assert class_num == cfg.MODEL.CLASSES, 'cfg.MODEL.CLASSES数值({})与训练文件夹下真实类别数({})不匹配'.format(cfg.MODEL.CLASSES, class_num)

    if not cfg.TRAIN.DATASETS or not os.path.exists(train_floder):
        print("请指定正确的训练集路径：TRAIN.DATASETS")
        return None

    if cfg.TRAIN.TRANSFORM == "AugMixAugment":
        ops = TRANSFORMS_REGISTRY.get(cfg.TRAIN.AUGMIX.OPS)()
        resize_trans = TRANSFORMS_REGISTRY.get(cfg.TRAIN.AUGMIX.RESIZE)(cfg.TRAIN.INPUT_WIDTH, cfg.TRAIN.INPUT_HEIGHT)
        train_transform = TRANSFORMS_REGISTRY.get(cfg.TRAIN.TRANSFORM)(ops, resize_trans)
    else:
        train_transform = TRANSFORMS_REGISTRY.get(cfg.TRAIN.TRANSFORM)(cfg.TRAIN.INPUT_WIDTH, cfg.TRAIN.INPUT_HEIGHT)

    train_dataset = ImageFolder(train_floder, train_transform)

    if cfg.MODEL.TYPE == 'cls':
        train_sampler = None
        shuffle = True
        if comm.get_world_size() > 1:
            shuffle = False
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = dataloader(train_dataset, batch_size=cfg.TRAIN.INPUT_BATCH, shuffle=shuffle, sampler=train_sampler, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True)
    else:
        data_sampler = BalancedIdentitySampler(train_dataset.targets, n_classes=cfg.TRAIN.METRIC.NUM_CLASSES, n_samples=cfg.TRAIN.METRIC.NUM_SAMPLES)
        batch_samplers = torch.utils.data.sampler.BatchSampler(data_sampler, cfg.TRAIN.METRIC.NUM_CLASSES * cfg.TRAIN.METRIC.NUM_SAMPLES, True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_samplers, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)

    val_path = os.path.join(cfg.TRAIN.DATASETS, "val")
    val_loader = create_infer_dataloader(val_path, cfg)

    return train_loader, val_loader