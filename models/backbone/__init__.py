from .resnet import build_ResNet101

def build_backbone(backbone, output_stride):
    if backbone == 'resnet':
        return build_ResNet101(output_stride=output_stride, pretrained=True)