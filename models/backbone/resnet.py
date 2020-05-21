import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url

class BottleNeck(nn.Module):
    def __init__(self, in_channles, out_channels, stride, dilation):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channles, out_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=stride,
                      padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU()

        if in_channles != out_channels[2]:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channles, out_channels[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels[2])
            )
            self.increase_dim = True
        else:
            self.increase_dim = False
        self._init_weight()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.increase_dim == True:
            x = self.downsample(x)

        y = x + y
        y = self.relu(y)

        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class ResNet(nn.Module):
    def __init__(self, block, block_nums, output_stride, pretrained):
        super(ResNet, self).__init__()
        if output_stride == 16:
            downsamples = [False, True, True, False]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            downsamples = [False, True, False, False]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.block = block

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, [64, 64, 256], downsample=downsamples[0], dilation=dilations[0], num=block_nums[0])
        self.layer2 = self._make_layer(256, [128, 128, 512], downsample=downsamples[1], dilation=dilations[1], num=block_nums[1])
        self.layer3 = self._make_layer(512, [256, 256, 1024], downsample=downsamples[2], dilation=dilations[2], num=block_nums[2])
        # self.layer4 = self._make_layer(1024, [512, 512, 2048], downsample=downsamples[3], dilation=dilations[3], num=block_nums[3])
        self.layer4 = self._make_MG_layer(1024, [512, 512, 2048], rate=dilations[3], MG=[1,2,4])
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feature = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def _make_layer(self, in_channels, out_channels, downsample, dilation, num):
        layers = []
        layers.append(self.block(in_channels, out_channels, stride=2 if downsample else 1, dilation=dilation))
        for i in range(num-1):
            layers.append(self.block(out_channels[-1], out_channels, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_MG_layer(self, in_channels, out_channels, rate, MG):
        layers = []
        layers.append(self.block(in_channels, out_channels, stride=1, dilation=rate * MG[0]))
        layers.append(self.block(out_channels[-1], out_channels, stride=1, dilation=rate * MG[1]))
        layers.append(self.block(out_channels[-1], out_channels, stride=1, dilation=rate * MG[2]))
        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrained_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def build_ResNet101(output_stride, pretrained=True):
    return ResNet(BottleNeck, [3, 4, 23, 3], output_stride, pretrained)


if __name__ == '__main__':
    import torch
    import torchviz
    x = torch.randn(8, 3, 512, 512)
    model = build_ResNet101(output_stride=8, pretrained=True)
    x, low_level_feature = model(x)
    torchviz.make_dot(x).render('ResNet', view=False)