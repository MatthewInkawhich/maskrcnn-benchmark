# MatthewInkawhich

"""
Switch modules for adaptive backbone models
"""
import torch
import torch.nn.functional as F
from torch import nn


################################################################################
### ILAdaptive Switch
################################################################################
class ILAdaptive_Switch(nn.Module):
    def __init__(self, in_channels, num_branches, out_channels_conv=256, intermediate_channels_linear=512):
        super(ILAdaptive_Switch, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels_conv, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels_conv, intermediate_channels_linear)
        self.fc2 = nn.Linear(intermediate_channels_linear, num_branches)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
