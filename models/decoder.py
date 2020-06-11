import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=int((kernel_size-1)/2), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

class Decoder(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Decoder, self).__init__()
        if backbone == 'resnet101' or backbone == 'resnet50':
            low_level_channels = 256
        else:
            raise NotImplementedError
        self.conv1 = ConvBlock(low_level_channels, 48, kernel_size=1, stride=1)
        self.last_conv = nn.Sequential(
            ConvBlock(304, 256, kernel_size=3, stride=1),
            nn.Dropout(0.5),
            ConvBlock(256, 256, kernel_size=3, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        x = F.interpolate(x, size=low_level_feature.shape[2:],mode='bilinear', align_corners=True)
        x = torch.cat([low_level_feature, x], dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

def build_decoder(backbone, num_classes):
    return Decoder(backbone, num_classes)

if __name__ == '__main__':
    import torchviz
    decoder = build_decoder('resnet', 10)
    low_level_feature = torch.randn(4, 256, 56, 56)
    x = torch.randn(4, 256, 14, 14)
    y = decoder(low_level_feature, x)
    torchviz.make_dot(y).render('decoder', view=False)