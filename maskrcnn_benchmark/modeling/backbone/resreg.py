# MatthewInkawhich

"""
Resolution regulator modules. Put after the resnet body in the "backbone" module to change
feature resolution right before RPN.
"""
import torch
import torch.nn.functional as F
from torch import nn


class Up4x(nn.Module):
    def __init__(self, cfg):
        super(Up4x, self).__init__()
        channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.regulator = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=4, padding=0)
    def forward(self, x):
        return [F.relu(self.regulator(x[0]))]


class Up2x(nn.Module):
    def __init__(self, cfg):
        super(Up2x, self).__init__()
        channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.regulator = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return [F.relu(self.regulator(x[0]))]


class Keep1x(nn.Module):
    def __init__(self, cfg):
        super(Keep1x, self).__init__()
        channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.regulator = nn.Conv2d(channels, channels, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        return [F.relu(self.regulator(x[0]))]


class Down2x(nn.Module):
    def __init__(self, cfg):
        super(Down2x, self).__init__()
        channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.regulator = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return [F.relu(self.regulator(x[0]))]
