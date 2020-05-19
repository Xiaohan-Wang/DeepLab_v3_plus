import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(_ASPPModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2*dilation),
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(256)
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


class _GlobalAvgPool(nn.Module):
    def __init__(self, in_channels):
        super(_GlobalAvgPool, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            # TODO: Q: bn would make each channel to 0 and add a bias, so what's the meaning of GAP?
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self._init_weight()

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.model(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()
        if backbone == 'resnet':
            in_channel = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        kernel_sizes = [1, 3, 3, 3]
        aspp_modules = []
        # atrous convolution
        for i in range(4):
            aspp_modules.append(_ASPPModule(in_channel, kernel_sizes[i], dilations[i]))
        # global average pooling
        aspp_modules.append(_GlobalAvgPool(in_channel))
        self.aspp_modules = nn.ModuleList(aspp_modules)
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # TODO: Q: why add dropout
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        y = []
        for i in range(len(self.aspp_modules)):
            y.append(self.aspp_modules[i](x))
        y = torch.cat(y, dim=1)
        y = self.dim_reduction(y)
        return self.dropout(y)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

def build_ASPP(backbone, output_stride):
    return ASPP(backbone, output_stride)



if __name__ == '__main__':
    import torchviz
    from torch.autograd import Variable
    aspp = build_ASPP(backbone='resnet', output_stride=16)
    # for name, m in aspp.named_children():
    #     print(name+':', m)
    x = Variable(torch.randn(8,2048, 16, 16))
    x = aspp(x)
    torchviz.make_dot(x).view()






