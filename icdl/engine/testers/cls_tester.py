import os
import torch
from tqdm import tqdm

from .tester import DefaultTester
from icdl.datasets.folder_dataset import ImageFolder
from icdl.datasets.registry import TRANSFORMS_REGISTRY
from icdl.engine.registry import ENFINE_REGISTRY


@ENFINE_REGISTRY.register()
class CLSTester(DefaultTester):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, *args, **kwargs):
        if not os.path.exists(self.cfg.TEST.DATAPATH):
            print("请检查cfg.TEST.DATAPATH路径")
            return

        transform = TRANSFORMS_REGISTRY.get(self.cfg.INFERENCE.TRANSFORM)(self.cfg.TRAIN.INPUT_WIDTH, self.cfg.TRAIN.INPUT_HEIGHT)
        dataset = ImageFolder(self.cfg.TEST.DATAPATH, transform)

        total_corrects = 0.0
        label_count = {}
        label_corrects = {}
        for img, label, path in tqdm(dataset):
            img = img.cuda()

            with torch.no_grad():
                output = self.model(img.unsqueeze(dim=0))

            _, pred = torch.max(output[0], 0)
            pred = pred.cpu().item()

            if pred not in label_count.keys():
                label_count[pred] = 0
                label_corrects[pred] = 0

            if pred == label:
                total_corrects += 1
                label_corrects[pred] += 1

            label_count[pred] += 1

        for label in label_count.keys():
            print('类别{}准确率：{:.6f} ({}/{})'.format(dataset.classes[label], float(label_corrects[label] / label_count[label]), int(label_corrects[label]), int(label_count[label])))

        print("测试集总体准确率: {:.6f} ({}/{})".format(total_corrects / len(dataset), total_corrects, len(dataset)))