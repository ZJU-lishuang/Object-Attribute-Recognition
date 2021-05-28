import torch
import torch.nn as nn
from model.backbones.resnet import ResNet
from model.heads.cls_head import AttrHead
import logging

logger = logging.getLogger(__name__)

class buildModel(nn.Module):
    def __init__(self,feat_dim=512,num_classes=1):
        super(buildModel, self).__init__()
        self.backbone=ResNet(18)
        self.head=AttrHead(feat_dim,num_classes)

    def forward(self,inputs):
        features=self.backbone(inputs)
        results=self.head(features)

        return results

