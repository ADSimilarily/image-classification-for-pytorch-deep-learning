from icdl.datasets.build import build_train_dataloader
from icdl.datasets.samplers.hard_sampler import HardSampler
from icdl.models.build_models import build_models
from icdl.solver.build import build_optimer, build_lr_scheduler
from icdl.loss.build import build_criterion
from icdl.utils.files import load_weight_without_strict
from icdl.utils import comm
from icdl.utils.logger import Logger

import torch
from torch.nn.parallel import DistributedDataParallel
try:
    import apex
except:
    print("没有 apex库")


class DefaultTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        assert self.cfg.MODEL.TYPE
        assert self.cfg.TRAIN.DATASETS
        assert self.cfg.OUTPUT.DIR
        Logger.init(cfg.OUTPUT.DIR, comm.get_local_rank())
        Logger.log(cfg)

        self.train_loader, self.val_loader = build_train_dataloader(cfg)
        Logger.log('训练集数据量：{}   验证集数据量：{}'.format(len(self.train_loader.dataset),
                                             (len(self.val_loader.dataset) if self.val_loader else 0)))
        self.model = build_models(cfg)
        self.load_weights()
        self.model = self.model.cuda()
        Logger.log(self.model)

        self.optimer = build_optimer(cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimer, len(self.train_loader.dataset))
        self.criterion = build_criterion(cfg)

        if cfg.TRAIN.APEX:
            self.model, self.optimer = apex.amp.initialize(self.model, self.optimer)
        if comm.get_world_size() > 1:
            if cfg.TRAIN.APEX:
                self.model = apex.parallel.convert_syncbn_model(self.model)
                self.model = apex.parallel.DistributedDataParallel(self.model)
            else:
                self.model = DistributedDataParallel(self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

    def __call__(self, *args, **kwargs):
        Logger.log("=======================START TRAIN============================")
        best_acc = 0.0
        for i in range(self.cfg.SOLVER.START_ITER, self.cfg.SOLVER.MAX_ITER):
            HardSampler.epoch(i)

            acc = self.train_one_epoch(i)
            if acc > best_acc and i != 0:
                best_acc = acc
                self.save_weights('train_best')

            if self.cfg.OUTPUT.SAVE_INTERVAL and i != 0 and i % self.cfg.OUTPUT.SAVE_INTERVAL == 0:
                self.val_one_epoch(i)
                self.save_weights(str(i))

    def train_one_epoch(self, epoch):
        return NotImplementedError

    def val_one_epoch(self, i):
        return NotImplementedError

    def save_weights(self, i):
        if comm.is_main_process():
            save_name = "%s/%s_%s_%s.pth" % (self.cfg.OUTPUT.DIR, self.cfg.MODEL.TYPE, self.cfg.MODEL.BACKBONE, i)

            if self.cfg.OUTPUT.SAVE_SUFFIX:
                save_name = save_name.replace(".pth", "_" + self.cfg.OUTPUT.SAVE_SUFFIX + ".pth")

            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), save_name)
            else:
                torch.save(self.model.state_dict(), save_name)

    def load_weights(self):
        if self.cfg.MODEL.WEIGHTS:
            pretrained_dict = torch.load(self.cfg.MODEL.WEIGHTS, map_location='cpu')

            if self.cfg.MODEL.STRICT:
                self.model.load_state_dict(pretrained_dict, strict=True)
            else:
                load_weight_without_strict(self.model, pretrained_dict)