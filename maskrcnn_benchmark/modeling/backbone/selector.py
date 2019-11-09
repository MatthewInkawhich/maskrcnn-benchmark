# MatthewInkawhich

"""
Switch modules for adaptive backbone models
"""
import torch
import torch.nn.functional as F
from torch import nn


################################################################################
### EWAdaptive Selector
################################################################################
class EWAdaptive_Selector(nn.Module):
    def __init__(self, in_channels, num_branches):
        super(EWAdaptive_Selector, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.selector = nn.Conv2d(in_channels, num_branches, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.selector(x)
        return x
