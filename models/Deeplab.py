import torch.nn as nn

class Deeplab(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # TODO: get_baseline_net
        self.baseline_net = build_backbone(backbone)
        # TODO: ASPP module
        self.aspp = build_ASPP()
        # TODO: decoder
        self.decoder = build_decoder()

    def forward(self, x):
        low_level_feature, x = self.baseline_net(x)
        x = self.aspp(x)
        x = self.decoder(low_level_feature, x)
        return x
