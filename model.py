import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

MODEL_REGISTRY = {
    'resnet50':   {'fn': lambda: resnet50(weights=ResNet50_Weights.DEFAULT), 'classifier': 'fc', 'in_features': 2048},
    'resnet18':   {'fn': lambda: resnet18(weights=ResNet18_Weights.DEFAULT), 'classifier': 'fc', 'in_features': 512},
    'densenet121':{'fn': lambda: densenet121(weights=DenseNet121_Weights.DEFAULT), 'classifier': 'classifier', 'in_features': 1024},
    'efficientnet_b0': {'fn': lambda: efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT), 'classifier': 'classifier.1', 'in_features': 1280},
    'mobilenet_v3_small': {'fn': lambda: mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT), 'classifier': 'classifier.3', 'in_features': 1024}
}

def _set_module(model, attr_path, new_module):
    parts = attr_path.split('.')
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)

class Net(nn.Module):
    def __init__(self, num_classes=100, arch='resnet50'):
        super().__init__()
        cfg = MODEL_REGISTRY[arch]
        self.model = cfg['fn']()
        self.classifier_attr = cfg['classifier']
        _set_module(self.model, self.classifier_attr, nn.Linear(cfg['in_features'], num_classes))

    def forward(self, x):
        return self.model(x)

def freeze_backbone(model):
    prefix = model.classifier_attr.split('.')[0]
    for name, param in model.model.named_parameters():
        if not name.startswith(prefix):
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.model.parameters():
        param.requires_grad = True