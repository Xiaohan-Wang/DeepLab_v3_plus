import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import itertools

from models.backbone import build_backbone
from models.aspp import build_ASPP
from models.decoder import build_decoder


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21):
        super().__init__()
        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = build_ASPP(backbone, output_stride)
        self.decoder = build_decoder(backbone, num_classes)
        self._init_weight()

    def forward(self, x):
        h, w = x.shape[2:]
        x, low_level_feature = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def get_1x_lr_parameters(self):
        return self.backbone.parameters()

    def get_10x_lr_parameters(self):
        params = [self.aspp.parameters(), self.decoder.parameters()]
        return itertools.chain(*params)

def build_Deeplab(backbone, output_stride, num_classes):
    return DeepLab(backbone, output_stride, num_classes)


if __name__ == '__main__':
    import torch
    import torchviz
    x = torch.randn(4, 3, 256, 256)
    model = build_Deeplab(backbone='resnet', output_stride=8, num_classes=14)
    y = model(x)
    print(y.shape)
    torchviz.make_dot(y).render('Deeplab', view=False)
