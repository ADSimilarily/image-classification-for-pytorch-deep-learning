#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch.nn as nn
from ..registry import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        return self.ce_loss(inputs, targets)
