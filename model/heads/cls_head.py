# encoding: utf-8
"""
@author:  lishuang
@contact: qqlishuang@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones.resnet import BasicBlock,Bottleneck
import logging

logger = logging.getLogger(__name__)

# class ClsHead(nn.Module):
#     arch_settings = {
#         18: (BasicBlock, (2, 2, 2, 2)),
#         34: (BasicBlock, (3, 4, 6, 3)),
#         50: (Bottleneck, (3, 4, 6, 3)),
#         101: (Bottleneck, (3, 4, 23, 3)),
#         152: (Bottleneck, (3, 8, 36, 3))
#     }
#     def __init__(self,depth,num_classes=1000):
#         super(ClsHead, self).__init__()
#         block, stage_blocks = self.arch_settings[depth]
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#     def forward(self,x):
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = F.softmax(self.fc(x))
#
#         return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class FastGlobalAvgPool(nn.Module):
    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class AttrHead(nn.Module):
    def __init__(self,feat_dim,num_classes):
        super().__init__()
        # num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.pool_layer=FastGlobalAvgPool()
        self.bottleneck=BatchNorm(feat_dim,bias_freeze=True)
        self.bottleneck.apply(weights_init_kaiming)

        self.weight = nn.Parameter(torch.normal(0, 0.01, (num_classes, feat_dim)))

        self.bnneck = nn.BatchNorm1d(num_classes)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, features):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)

        logits = F.linear(neck_feat, self.weight)
        logits = self.bnneck(logits)

        # Evaluation
        if not self.training:
            cls_outptus = torch.sigmoid(logits)
            return cls_outptus

        return {
            "cls_outputs": logits,
        }
    


# def build_resnet_head():
#     head = ClsHead(18)
#     pretrain = True
#     pretrain_path = "../../weights/resnet18-f37072fd.pth"
#     if pretrain:
#         if pretrain_path:
#             try:
#                 state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
#                 logger.info(f"Loading pretrained head from {pretrain_path}")
#             except FileNotFoundError as e:
#                 logger.info(f'{pretrain_path} is not found! Please check this path.')
#                 raise e
#             except KeyError as e:
#                 logger.info("State dict keys error! Please check the state dict.")
#                 raise e
#
#         incompatible = head.load_state_dict(state_dict, strict=False)
#
#     return head
#
# if __name__ == '__main__':
#     backbone=build_resnet_head()