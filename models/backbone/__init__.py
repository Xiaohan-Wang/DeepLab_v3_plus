from .resnet import build_ResNet101, build_ResNet50

def build_backbone(backbone, output_stride):
    if backbone == 'resnet101':
        return build_ResNet101(output_stride=output_stride, pretrained=True)
    elif backbone == 'resnet50':
        return build_ResNet50(output_stride=output_stride, pretrained=True)