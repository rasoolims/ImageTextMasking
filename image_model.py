import torch
import torch.nn.functional as F
from torchvision import models

class ModifiedResnet(models.ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        input = x
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        grid_hidden = self.layer4(x)

        return grid_hidden.permute((0,2,3,1))


def init_net(freeze:bool=False):
    model = models.resnet50(pretrained=True)
    model.__class__ = ModifiedResnet
    if freeze:
        model.eval()
    return model