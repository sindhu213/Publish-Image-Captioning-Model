import torch.nn as nn
from torchvision import models

class Resnet101(nn.Module):
  def __init__(self):
    super(Resnet101, self).__init__()
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    modules = list(resnet.children())[:-2]
    self.resnet = nn.Sequential(*modules)

    for params in self.resnet.parameters():
      params.requires_grad = False


  def forward(self,images):
    # images = [batch_size, 3, 224, 224]

    # encoder_out = [batch_size, 2048, 7, 7]
    encoder_out = self.resnet(images)
    return encoder_out