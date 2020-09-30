import torch
import torch.nn as nn

from ...utils import comm
from ..registry import LOSSES_REGISTRY
from ..semantic_agumentation import ISEALayer


@LOSSES_REGISTRY.register()
class CircleLoss(nn.Module):
    def __init__(self, memory_size=1, isea=False, scale=32, m=0.25):
        super(CircleLoss, self).__init__()
        self.memory_size = memory_size
        self.scale = scale
        self.m = m
        self.margin_pos = 1 - m
        self.margin_neg = m
        self.optim_pos = 1 + m
        self.optim_neg = -m
        self.theta = 0.25
        self.memory = None
        self.labels = None
        self.template = {}
        self.isea = ISEALayer(iter=1000, alpha=1.4142) if isea else None

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

        #计算近似模板
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

        if self.isea:
            noise_embeddings = self.isea(embeddings, labels)
            sim_mat = torch.matmul(noise_embeddings, torch.t(all_embedding))
        else:
            sim_mat = torch.matmul(embeddings, torch.t(all_embedding))

        loss = torch.zeros(1, device=embeddings.device)
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

            sim_pos_ = anchor[pos_condition]
            sim_neg_ = anchor[neg_condition]

            pos_min = sim_pos_.min()
            neg_max = sim_neg_.max()

            if pos_min > neg_max:
                low_match += 1
                if pos_min > self.margin_pos:
                    medium_match += 1
                if pos_min > self.margin_pos and neg_max < self.margin_neg:
                    high_match += 1

            sim_neg = sim_neg_[sim_neg_ - self.theta > self.margin_neg]
            sim_pos = sim_pos_[sim_pos_ + self.theta < self.margin_pos]

            if sim_neg.size(0) == 0:
                sim_neg = sim_neg_[sim_neg_ > self.margin_neg]
                if sim_neg.size(0) == 0:
                    sim_neg = sim_neg_.max()
            if sim_pos.size(0) == 0:
                sim_pos = sim_pos_[sim_pos_ < self.margin_pos]
                if sim_pos.size(0) == 0:
                    sim_pos = sim_pos_.min()

            with torch.no_grad():
                alpha_pos = torch.clamp(-sim_pos + self.optim_pos, min=0)
                alpha_neg = torch.clamp(sim_neg - self.optim_neg, min=0)

            item_pos = -self.scale * alpha_pos * (sim_pos - self.margin_pos)
            item_neg = self.scale * alpha_neg * (sim_neg - self.margin_neg)

            se_pos = torch.logsumexp(item_pos, -1)
            se_neg = torch.logsumexp(item_neg, -1)
            if se_pos + se_neg > 80.0:
                current_loss = se_pos + se_neg
            else:
                current_loss = torch.log1p(torch.exp(se_pos + se_neg))

            loss += current_loss

        return loss/batch_size, low_match/batch_size, medium_match/batch_size, high_match/batch_size