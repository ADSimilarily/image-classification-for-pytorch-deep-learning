import os
import shutil
import torch
from icdl.models.build_models import build_models


class DefaultTester:
    def __init__(self, cfg):
        self.cfg = cfg
        weight_path = cfg.MODEL.WEIGHTS
        if weight_path.endswith('.pth'):
            self.model = build_models(cfg)
            self.model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS, map_location='cpu'), strict=True)
        else:
            self.model = torch.jit.load(cfg.MODEL.WEIGHTS, map_location='cpu')
        self.model.eval().cuda()

        if self.cfg.TEST.OUTPATH:
            if not os.path.exists(self.cfg.TEST.OUTPATH):
                os.mkdir(self.cfg.TEST.OUTPATH)

    def __call__(self, *args, **kwargs):
        pass

    def copy_img_to_output_path(self, label, path):
        dir = os.path.join(self.cfg.TEST.OUTPATH, str(label))
        if not os.path.exists(dir):
            os.mkdir(dir)

        shutil.copy(path, dir)