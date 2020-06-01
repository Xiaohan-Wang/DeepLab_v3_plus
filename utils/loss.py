import torch.nn as nn
import torch


class SegmentationLoss():
    def __init__(self, weight=None, ignore_index=255):
        # weight should be a tensor
        self.weight = weight
        self.ignore_index = ignore_index

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'weighted ce':
            return self.WeightedCrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'weighted focal':
            return self.WeightedFocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, input, target):
        if self.weight is not None:
            raise ValueError("Received weight not equal to None. Try WeightedCrossEntropyLoss instead.")
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        loss = criterion(input, target)
        return loss

    def WeightedCrossEntropyLoss(self, input, target):
        if self.weight is None:
            raise ValueError("Weight should be a list. Got None")
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        loss = criterion(input, target)
        return loss

    def FocalLoss(self, input, target):
        if self.weight is not None:
            raise ValueError("Received weight not equal to None. Try WeightedFocalLoss instead.")
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        minus_log_pt = criterion(input, target)
        pt = torch.exp(-minus_log_pt)
        scale = (1 - pt) ** 2
        loss = scale * minus_log_pt
        loss = loss.sum()/scale.sum()
        return loss

    def WeightedFocalLoss(self, input, target):
        if self.weight is None:
            raise ValueError("Weight should be a list. Got None")
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        minus_log_pt = criterion(input, target)
        pt = torch.exp(-minus_log_pt)
        weightt = self.weight[target]
        scale = (1 - pt) ** 2 * weightt
        loss = scale * minus_log_pt
        loss = loss.sum() / scale.sum()
        return loss



if __name__ == '__main__':
    import numpy as np
    weight = torch.Tensor([0.3,0.8])
    target = torch.Tensor([[[1, 0]]]).long()
    print(target.shape)
    input = torch.Tensor([[[[1,-1]],[[2,0]]]])
    print(input.shape)
    criterion = SegmentationLoss(weight=weight).build_loss(mode='weighted ce')
    loss = criterion(input,target)
    print(loss)

    # correct answer: pay attention to the weighted mean
    # 'ce': 0.813
    # 'weighted cw': 0.586
    # 'focal': 1.194
    # 'weighted focal': 1.048




