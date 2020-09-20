import os
import torch
import cv2
from tqdm import tqdm
from .tester import DefaultTester
from icdl.engine.registry import ENFINE_REGISTRY
from icdl.datasets.registry import TRANSFORMS_REGISTRY
from icdl.datasets.folder_dataset import ImageFolder
from icdl.datasets.build import create_infer_dataloader


@ENFINE_REGISTRY.register()
class METRICUnlabelTester(DefaultTester):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, *args, **kwargs):
        if not os.path.exists(self.cfg.TEST.DATAPATH) or not os.path.exists(self.cfg.TEST.METRIC.TEMPLATE_PATH):
            print("请检查cfg.TEST.DATAPATH和cfg.TEST.METRIC.TEMPLATE_PATH路径")
            return

        template = torch.load(self.cfg.TEST.METRIC.TEMPLATE_PATH, map_location='cpu')
        embeddings = torch.zeros((1, self.cfg.MODEL.METRIC.DIM)).cuda()
        labels = []
        for key, value in template.items():
            labels.append(key)
            embeddings = torch.cat((embeddings, value.cuda().unsqueeze(0)), 0)
        embeddings = embeddings[1:]

        transform = TRANSFORMS_REGISTRY.get(self.cfg.INFERENCE.TRANSFORM)(self.cfg.TRAIN.INPUT_WIDTH, self.cfg.TRAIN.INPUT_HEIGHT)
        features = None
        paths = []

        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        files = os.listdir(self.cfg.TEST.DATAPATH)
        files = [file for file in files if file.endswith(IMG_EXTENSIONS)]
        for file in tqdm(files):
            path = os.path.join(self.cfg.TEST.DATAPATH, file)
            sample = cv2.imread(path)
            image = transform(image=sample)['image']
            with torch.no_grad():
                feature = self.model(image.cuda())

            paths.append(path)
            if features is None:
                features = feature
            else:
                features = torch.cat([features, feature], dim=0)

        simi = torch.matmul(features, torch.t(embeddings))
        sorted_simi, sorted_idx = torch.sort(simi, 1, True)

        for idx in range(sorted_idx.shape[0]):
            print(f"文件名({paths[idx]})  预测类别：{labels[sorted_idx[idx][0].cpu().numpy()]}"
                  f"({sorted_simi[idx][0].cpu().numpy()})   第二相似类别：{labels[sorted_idx[idx][1].cpu().numpy()]}({sorted_simi[idx][1].cpu().numpy()})")

            if self.cfg.TEST.OUTPATH:
                self.copy_img_to_output_path(labels[sorted_idx[idx][0].cpu().numpy()], paths[idx])


@ENFINE_REGISTRY.register()
class METRICTester(DefaultTester):
    def __init__(self, cfg):
        super().__init__(cfg)

    def test_one_sub_folder(self, features, target, embeddings, labels):
        simi = torch.matmul(features, torch.t(embeddings))

        #计算当前标签类别的平均相似度，值越大越好
        p = labels.index(target)
        first_simi = torch.mean(simi[:, p])

        #计算最大相似度对应的类别是不是目标类别，是则对，不是则错
        _, hard_index = simi.max(1)
        label_list = [labels[i] for i in hard_index.cpu().numpy()]
        ok = 0.0
        ng = 0.0
        error_label = ''
        for idx, label in enumerate(label_list):
            if label == target:
                ok += 1
            else:
                ng += 1
                error_label += (label + ' ')

        #计算第二相似类别的相似度，值越小越好
        sorted_simi, sorted_idx = torch.sort(simi, 1, True)
        second_simi = torch.mean(sorted_simi[:, 1])

        return ok, ng, first_simi, second_simi, error_label

    def __call__(self, *args, **kwargs):
        if not os.path.exists(self.cfg.TEST.DATAPATH) or not os.path.exists(self.cfg.TEST.METRIC.TEMPLATE_PATH):
            print("请检查cfg.TEST.DATAPATH和cfg.TEST.METRIC.TEMPLATE_PATH路径")
            return

        dataloader = create_infer_dataloader(self.cfg.TEST.DATAPATH, self.cfg)

        labels = None
        features = None
        features_map = {}
        for image, target in tqdm(dataloader):
            with torch.no_grad():
                feature = self.model(image.cuda())

            if labels is None:
                labels = target
                features = feature
            else:
                labels = torch.cat([labels, target], dim=0)
                features = torch.cat([features, feature], dim=0)

        labels_set = set(labels.numpy())
        for label in labels_set:
            pos = labels == label
            target_embeddings = features[pos]
            features_map[dataloader.dataset.classes[label]] = target_embeddings

        template = torch.load(self.cfg.TEST.METRIC.TEMPLATE_PATH, map_location='cpu')
        embeddings = torch.zeros((1, self.cfg.MODEL.METRIC.DIM)).cuda()
        targets = []
        for key, value in template.items():
            targets.append(key)
            embeddings = torch.cat((embeddings, value.cuda().unsqueeze(0)), 0)
        embeddings = embeddings[1:]

        total_ok = 0.0
        total_ng = 0.0
        total_first_sim = 0.0
        total_second_sim = 0.0
        for key, value in features_map.items():
            if key not in targets:
                print("类别{}不在模板中".format(key))
                continue

            ok, ng, first_sim, second_sim, error_label = self.test_one_sub_folder(value, key, embeddings, targets)
            print("类别{}，数量{}————错误率：{}  错误数：{} 目标类别相似度：{} 最大类间相似度：{}  错误标签：{}".format(
                key, ok + ng, ng / (ok + ng), ng, first_sim, second_sim, error_label))

            total_ok += ok
            total_ng += ng
            total_first_sim += first_sim * (ok + ng)
            total_second_sim += second_sim * (ok + ng)

        print("测试完成,总数{}————错误率：{}  错误数：{} 目标类别相似度：{}  最大类间相似度:{}".format(
            total_ok + total_ng, total_ng / (total_ok + total_ng), total_ng,
            total_first_sim / (total_ok + total_ng), total_second_sim / (total_ok + total_ng)))


@ENFINE_REGISTRY.register()
class METRICClusterTester(DefaultTester):
    def __init__(self, cfg):
        super().__init__(cfg)

    def test_one_image(self, feature, embeddings, labels):
        simi = torch.matmul(feature, torch.t(embeddings))
        sorted_simi, sorted_idx = torch.sort(simi, 1, True)

        return labels[sorted_idx[0, 0]], sorted_simi[0, 0]

    def __call__(self, *args, **kwargs):
        if not os.path.exists(self.cfg.TEST.DATAPATH) or not os.path.exists(self.cfg.TEST.METRIC.TEMPLATE_PATH):
            print("请检查cfg.TEST.DATAPATH和cfg.TEST.METRIC.TEMPLATE_PATH路径")
            return

        transform = TRANSFORMS_REGISTRY.get(self.cfg.INFERENCE.TRANSFORM)(self.cfg.TRAIN.INPUT_WIDTH, self.cfg.TRAIN.INPUT_HEIGHT)
        dataset = ImageFolder(self.cfg.TEST.DATAPATH, transform)

        template = torch.load(self.cfg.TEST.METRIC.TEMPLATE_PATH, map_location='cpu')
        labels = template['labels']
        embeddings = template['features'].cuda()

        ok = 0.0
        ng = 0.0
        fail = []
        for image, target in tqdm(dataset):
            target = dataset.classes[target]
            with torch.no_grad():
                feature = self.model(image.unsqueeze(dim=0).cuda())

            label, simi = self.test_one_image(feature, embeddings, labels)

            if label is not None:
                if label == target:
                    ok += 1
                else:
                    ng += 1
                    fail.append((simi, target, label))

        for simi, target, label in fail:
            print('{}  {}:{} '.format(simi, target, label))
        total = ok + ng
        print('总计{}数据，其中错误{}个，占比{}'.format(total, ng, ng / total))