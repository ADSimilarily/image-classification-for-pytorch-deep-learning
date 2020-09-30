import torch
import torch.nn as nn

from ...utils import comm
from ..registry import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class MSOptimLoss(nn.Module):
    def __init__(self, memory_size=1, scale_pos=2.0, scale_neg=40.0, m=0.25):
        super(MSOptimLoss, self).__init__()
        self.m = m
        self.tpos = 1 - m
        self.tneg = m
        self.optim_pos = 1 + m
        self.optim_neg = -m
        self.margin = self.tpos - self.tneg

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

        self.memory_size = memory_size
        self.memory = None
        self.labels = None
        self.template = {}

    def getApproximateTemplate(self):
        return self.template

    def forward(self, embeddings, labels):
        assert embeddings.size(0) == labels.size(0), \
            f"feats.size(0): {embeddings.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = embeddings.size(0)

        if comm.get_world_size() > 1:
            all_embedding = comm.concat_all_gather(embeddings)
            all_targets = comm.concat_all_gather(labels)
        else:
            all_embedding = embeddings
            all_targets = labels

        # 计算近似模板
        targets_set = set(all_targets.cpu().numpy())
        for target in targets_set:
            pos = all_targets == target
            target_embeddings = all_embedding[pos]
            mean_embedding = target_embeddings.mean(dim=0)
            denom = mean_embedding.norm(2, 0, True).clamp_min(1e-12)
            mean_embedding = mean_embedding / denom
            self.template[target] = mean_embedding.unsqueeze(dim=0)

        if self.memory_size > all_embedding.shape[0]:
            if self.memory is None:
                self.memory = all_embedding
                self.labels = all_targets
            elif self.memory.shape[0] < self.memory_size:
                self.memory = torch.cat([all_embedding, self.memory], dim=0).detach()
                self.labels = torch.cat([all_targets, self.labels], dim=0).detach()
            else:
                drop_size = all_embedding.shape[0] + (self.memory.shape[0] - self.memory_size)
                splits = torch.split(self.memory, [self.memory.shape[0]-drop_size, drop_size], dim=0)
                split_labels = torch.split(self.labels, [self.memory.shape[0]-drop_size, drop_size], dim=0)
                self.memory = torch.cat([all_embedding, splits[0]], dim=0).detach()
                self.labels = torch.cat([all_targets, split_labels[0]], dim=0).detach()

            all_embedding = self.memory
            all_targets = self.labels

        sim_mat = torch.matmul(embeddings, torch.t(all_embedding))

        loss_ms = torch.zeros(1, device=embeddings.device)

        #仅满足pos_min > neg_max条件
        low_match = 0.0
        #上述条件基础上，额外满足pos_min > 0.75
        medium_match = 0.0
        #在满足pos_min > neg_max条件的基础上，额外满足
        high_match = 0.0

        for i in range(batch_size):
            anchor = sim_mat[i]   # shape: 1 × m
            pos_this = all_targets == labels[i]
            pos_condition = torch.zeros_like(anchor, dtype=torch.bool, device=labels.device)
            pos_condition[:all_targets.shape[0]][pos_this == True] = True

            neg_condition = torch.ones_like(anchor, dtype=torch.bool, device=labels.device)
            neg_condition[:all_targets.shape[0]][pos_this == True] = False

            pos_pair_ = anchor[pos_condition]
            pos_pair_ = pos_pair_.clamp(max=1.0)
            neg_pair_ = anchor[neg_condition]  # labels != labels[i]

            if pos_pair_.min() > neg_pair_.max():
                low_match += 1
                if pos_pair_.min() > self.tpos:
                    medium_match += 1
                if pos_pair_.min() > self.tpos and neg_pair_.max() < self.tneg:
                    high_match += 1

            neg_pair = neg_pair_[neg_pair_ + self.margin > pos_pair_.min()]
            pos_pair = pos_pair_[pos_pair_ - self.margin < neg_pair_.max()]

            if neg_pair.size(0) == 0:
                neg_pair = neg_pair_[neg_pair_ > self.tneg]
                if neg_pair.size(0) == 0:
                    neg_pair = neg_pair_.max()
            if pos_pair.size(0) == 0:
                pos_pair = pos_pair_[pos_pair_ < self.tpos]
                if pos_pair.size(0) == 0:
                    pos_pair = pos_pair_.min()

            with torch.no_grad():
                alpha_pos = torch.clamp(-pos_pair + self.optim_pos, min=0)
                alpha_neg = torch.clamp(neg_pair - self.optim_neg, min=0)

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log1p(torch.exp(torch.logsumexp(-self.scale_pos * alpha_pos * (pos_pair - self.tpos), -1)))
            neg_loss = 1.0 / self.scale_neg * torch.log1p(torch.exp(torch.logsumexp(self.scale_neg * alpha_neg * (neg_pair - self.tneg), -1)))
            loss_ms += pos_loss + neg_loss

        return loss_ms/batch_size, low_match/batch_size, medium_match/batch_size, high_match/batch_size