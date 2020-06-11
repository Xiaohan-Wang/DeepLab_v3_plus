import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, input, target):
        loss = self.criterion(input, target.long())
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight, ignore_index=-100):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')

    def forward(self, input, target):
        loss = self.criterion(input, target.long())
        return loss


class FocalLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, input, target):
        minus_log_pt = self.criterion(input, target.long())
        pt = torch.exp(-minus_log_pt)
        scale = (1 - pt) ** 2
        loss = scale * minus_log_pt
        loss = loss.sum() / scale.sum()
        return loss


class WeightedFocalLoss(nn.Module):
    def __init__(self, weight, ignore_index=-100):
        super(WeightedFocalLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, input, target):
        minus_log_pt = self.criterion(input, target.long())
        pt = torch.exp(-minus_log_pt)
        weightt = self.weight[target]
        scale = (1 - pt) ** 2 * weightt
        loss = scale * minus_log_pt
        loss = loss.sum() / scale.sum()
        return loss


def build_loss(mode="Focal"):
    if mode=='CrossEntropy':
        return CrossEntropyLoss
    elif mode=='WeightedCrossEntropy':
        return WeightedCrossEntropyLoss
    elif mode=='Focal':
        return FocalLoss
    elif mode=='WeightedFocal':
        return WeightedFocalLoss
    else:
        raise NotImplementedError


if __name__ == '__main__':
    weight = torch.Tensor([0.3,0.8])
    target = torch.Tensor([[[1, 0]]]).long()
    print(target.shape)
    input = torch.Tensor([[[[1,-1]],[[2,0]]]])
    print(input.shape)
    criterion = FocalLoss()
    loss = criterion(input,target)
    print(loss)

    # correct answer: pay attention to the weighted mean
    # 'ce': 0.813
    # 'weighted ce': 0.586
    # 'focal': 1.194
    # 'weighted focal': 1.048




