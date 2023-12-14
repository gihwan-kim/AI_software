import torch
import math
import torch.nn as nn

__all__ = ['CrossEntropyLoss', ]

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self,
                 weight=None,
                 ignore_index=-1):

        super(CrossEntropyLoss, self).__init__(weight, None, ignore_index)

    def forward(self, pred, target):
            return super(CrossEntropyLoss, self).forward(pred, target)
