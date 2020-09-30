import torch
from icdl.models.build_models import build_models
from icdl.datasets.build import create_infer_dataloader
from .registry import ENFINE_REGISTRY
from icdl.datasets.registry import TRANSFORMS_REGISTRY
from icdl.datasets.folder_dataset import ImageFolder
import os
from tqdm import tqdm


@ENFINE_REGISTRY.register()
def tmp(cfg):

    weight_path = cfg.MODEL.WEIGHT
    if weight_path.endswith('.pth'):
        model = build_models(cfg)
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location='cpu'), strict=True)
    else:
        model = torch.jit.load(cfg.MODEL.WEIGHT, map_location='cpu')
    model.eval().cuda()

    floder = cfg.TEMPLATE.DATAPATH
    if not cfg.TEMPLATE.DATAPATH or not os.path.exists(floder):
        print("请指定正确的数据集路径：TEMPLATE.DATASETS")
        return None

    dataloader = create_infer_dataloader(floder, cfg)

    labels = None
    features = None
    features_template = {}
    for image, target, _ in tqdm(dataloader):
        with torch.no_grad():
            feature = model(image.cuda())

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
        features_template[dataloader.dataset.classes[label]] = target_embeddings

    for k in features_template.keys():
        feature = features_template[k].mean(dim=0)

        denom = feature.norm(2, 0, True).clamp_min(1e-12)
        feature = feature / denom

        features_template[k] = feature

    torch.save(features_template, os.path.join(cfg.OUTPUT.DIR, os.path.basename(cfg.MODEL.WEIGHT).split('.')[0] + '_features_template.pth'))



#=======================================使用阈值聚类方法类做模板===================================

#单次聚类结果
def find_same(valid_set, simi, threshold):
    cluster_list = [valid_set.pop()]
    for anchor in cluster_list:
        for index, sim in enumerate(simi[anchor]):
            if index not in cluster_list and sim.cpu().item() > threshold:
                cluster_list.append(index)
    return cluster_list

#聚类，方法为随机选择种子，将与种子相似度大于阈值的加入列表，归为一类。依次用列表中的其他元素作为种子，直至便利完毕，列表不再变化
def cluster(features, threshold = 0.98):
    simi = torch.matmul(features, torch.t(features))

    #有效集合最开始为该类别的全集
    valid_set = { i for i in range(simi.shape[0])}
    result_list = []

    #若有效集合不为空，则表面聚类没有完成
    while valid_set:
        #返回本次聚类结果
        cluster_set = set(find_same(valid_set, simi, threshold))

        result_list.append(cluster_set)
        #更新有效集合，删除已聚类成功的
        valid_set = valid_set.difference(cluster_set)

    cluster_num = len(result_list)
    feature_list = []
    for single_set in result_list:
        indices = torch.tensor(list(single_set)).cuda()
        feature = torch.index_select(features, 0, indices)

        feature = feature.mean(dim=0)
        denom = feature.norm(2, 0, True).clamp_min(1e-12)
        feature = feature / denom
        feature_list.append(feature.unsqueeze(dim=0))

    return cluster_num, torch.cat(feature_list, dim=0)


@ENFINE_REGISTRY.register()
def cluster_tmp(cfg):
    weight_path = cfg.MODEL.WEIGHT
    if weight_path.endswith('.pth'):
        model = build_models(cfg)
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location='cpu'), strict=True)
    else:
        model = torch.jit.load(cfg.MODEL.WEIGHT, map_location='cpu')
    model.eval().cuda()

    floder = cfg.TEMPLATE.DATAPATH
    if not cfg.TEMPLATE.DATAPATH or not os.path.exists(floder):
        print("请指定正确的数据集路径：TEMPLATE.DATASETS")
        return None

    transform = TRANSFORMS_REGISTRY.get(cfg.INFERENCE.TRANSFORM)(cfg.TRAIN.INPUT_WIDTH, cfg.TRAIN.INPUT_HEIGHT)
    dataset = ImageFolder(floder, transform)

    cluster_map = {}
    for image, target, _ in tqdm(dataset):
        target = dataset.classes[target]
        with torch.no_grad():
            features = model(image.unsqueeze(dim=0).cuda())

        if target in cluster_map.keys():
            cluster_map[target] = torch.cat((cluster_map[target], features), dim=0)
        else:
            cluster_map[target] = features


    features_template = {}
    features_template['labels'] = []
    features_template['features'] = torch.zeros((1, cfg.MODEL.METRIC.DIM)).cuda()
    for key, value in cluster_map.items():
        #对单一类别聚类，返回聚类中心数量和聚类中心特征
        cluster_num, features = cluster(cluster_map[key])

        for i in range(cluster_num):
            features_template['labels'].append(key)
        features_template['features'] = torch.cat((features_template['features'], features), dim=0)

    features_template['features'] = features_template['features'][1:]
    print(features_template['features'].shape)
    torch.save(features_template, os.path.join(cfg.OUTPUT.DIR, os.path.basename(cfg.MODEL.WEIGHT).split('.')[0] + '_features_template.pth'))


#=======================================对每一个实例做模板===================================
@ENFINE_REGISTRY.register()
def instance_tmp(cfg):

    weight_path = cfg.MODEL.WEIGHT
    if weight_path.endswith('.pth'):
        model = build_models(cfg)
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location='cpu'), strict=True)
    else:
        model = torch.jit.load(cfg.MODEL.WEIGHT, map_location='cpu')
    model.eval().cuda()

    floder = cfg.TEMPLATE.DATAPATH
    if not cfg.TEMPLATE.DATAPATH or not os.path.exists(floder):
        print("请指定正确的数据集路径：TEMPLATE.DATASETS")
        return None

    transform = TRANSFORMS_REGISTRY.get(cfg.INFERENCE.TRANSFORM)(cfg.TRAIN.INPUT_WIDTH, cfg.TRAIN.INPUT_HEIGHT)
    dataset = ImageFolder(floder, transform)

    features_template = {}
    features_template['labels'] = []
    features_template['paths'] = []
    features_template['features'] = torch.zeros((1, cfg.MODEL.METRIC.DIM)).cuda()
    for image, target, path in tqdm(dataset):
        target = dataset.classes[target]
        with torch.no_grad():
            features = model(image.unsqueeze(dim=0).cuda())

        simi = torch.matmul(features, torch.t(features_template['features']))
        sorted_simi, sorted_idx = torch.sort(simi, 1, True)

        if (sorted_simi[0, 0].cpu().item() < 0.99):
            features_template['labels'].append(target)
            features_template['paths'].append(path)
            features_template['features'] = torch.cat((features_template['features'], features), dim=0)

    features_template['features'] = features_template['features'][1:]
    print(features_template['features'].shape)
    torch.save(features_template, os.path.join(cfg.OUTPUT.DIR, os.path.basename(cfg.MODEL.WEIGHT).split('.')[0] + '_features_template.pth'))