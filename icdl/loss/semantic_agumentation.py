import torch
import torch.nn as nn


class EstimatorCV(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(EstimatorCV, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num

    def forward(self, features, labels):
        N = features.size(0)
        C = self.class_num
        D = self.feature_dim

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)   # 根据label生成onehot编码
        batch_class_onehot = onehot.unsqueeze(-1).expand(N, C, D)       # 将onehot在dim维度扩充

        # 统计batch中,各个类别的数量, 求均值和方差的时候使用
        amount_class_dim = batch_class_onehot.sum(0)
        amount_class_dim[amount_class_dim == 0] = 1  # 防止分母为0

        # 根据扩充后的onehot,提取batch各自类别的feature
        batch_class_features = features.unsqueeze(1).expand(N, C, D)  # 将feature在类别维度扩充
        extracted_batch_class_features = batch_class_features.mul(batch_class_onehot)

        # 1.求均值: 当前batch各自类别所在特征维度的均值
        ave_class_dim = extracted_batch_class_features.sum(0) / amount_class_dim

        # 2.求方差: 当前batch各自类别所在特征维度的方差
        var_class_dim = extracted_batch_class_features - ave_class_dim.expand(N, C, D).mul(batch_class_onehot)
        var_class_dim = var_class_dim.pow(2).sum(0).div(amount_class_dim)
        std_class_dim = var_class_dim.sqrt()[labels]
        return std_class_dim


class ISEALayer(nn.Module):
    def __init__(self, iter=1000, alpha=1.0):
        super(ISEALayer, self).__init__()
        self.iter = iter
        self.count = 1
        self.alpha = alpha

    def forward(self, features, labels):
        with torch.no_grad():
            noise = torch.randn(features.shape, dtype=features.dtype, device=features.device)

        covar = torch.zeros_like(features)
        labels_set = list(set(labels.cpu().numpy()))
        for label in labels_set:
            index = labels == label
            f = features[index]
            covar[index] = f.std(0)

        ratio = self.count / self.iter
        ratio = min(ratio, self.alpha)
        self.count += 1

        noise_features = covar * noise
        features = features + ratio * noise_features

        denom = features.norm(2, 1, True).clamp_min(1e-12)
        features = features / denom

        return features


if __name__ == "__main__":
    device = torch.device("cuda:0")
    es = ISEALayer()
    features = torch.rand((64, 2048)).to(device)
    label = torch.randint(0, 5, (64,)).to(device)
    es(features, label)