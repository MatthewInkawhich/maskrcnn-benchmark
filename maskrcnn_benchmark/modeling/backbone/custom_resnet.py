# Matthew Inkawhich

"""
Define a custom ResNet backbone module.
"""
from collections import namedtuple
import math

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry


################################################################################
### Helpers
################################################################################
# Convert scalar pad to tuple pad (left, right, top, bottom)
# Note: scalar padding represents how much padding to add on all sides
# In .5 cases, favor right, bottom
# Ex: scalar_pad=3.5 --> (3, 4, 3, 4)
def get_pad_tuple(scalar_pad):
    left = math.floor(scalar_pad)
    right = math.ceil(scalar_pad)
    return (left, right, left, right)



################################################################################
### CustomResNet Backbone Module
################################################################################
class CustomResNet(nn.Module):
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs under cfg.MODEL.CUSTOM_RESNET
        """
        super(CustomResNet, self).__init__()
        # Assert correct config format
        assert (len(cfg.MODEL.CUSTOM_RESNET.STEM_CONFIG) == len(cfg.MODEL.CUSTOM_RESNET.STEM_CHANNELS)), "Stem config must equal stem channels"
        assert (len(cfg.MODEL.CUSTOM_RESNET.BODY_CONFIG) == len(cfg.MODEL.CUSTOM_RESNET.BODY_CHANNELS)), "Body config must equal body channels"

        # Construct Stem
        self.C1 = Stem(cfg.MODEL.CUSTOM_RESNET.STEM_CONFIG, cfg.MODEL.CUSTOM_RESNET.STEM_CHANNELS)

        # Construct Stages
        self.C2 = Stage(cfg.MODEL.CUSTOM_RESNET.BODY_CONFIG[0], cfg.MODEL.CUSTOM_RESNET.BODY_CHANNELS[0])
        self.C3 = Stage(cfg.MODEL.CUSTOM_RESNET.BODY_CONFIG[1], cfg.MODEL.CUSTOM_RESNET.BODY_CHANNELS[1])
        self.C4 = Stage(cfg.MODEL.CUSTOM_RESNET.BODY_CONFIG[2], cfg.MODEL.CUSTOM_RESNET.BODY_CHANNELS[2])


    def forward(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs ([Tensor]): Final output feature map
        """
        outputs = []
        
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        outputs.append(x)
        return outputs



################################################################################
### Stem
################################################################################
class Stem(nn.Module):
    """
    Stem module
    Use group norm
    """
    def __init__(self, stem_config, stem_channels):
        super(Stem, self).__init__()

        # Initialize layers
        layers = []

        # Iterate over stem_config, build stem
        in_channels = 3
        for i in range(len(stem_channels)):
            # Initialize padding
            pad_tuple = get_pad_tuple(stem_config[i][2])
            layers.append(nn.ZeroPad2d(pad_tuple))

            # Initialize layer
            conv = Conv2d(in_channels, stem_channels[i], kernel_size=stem_config[i][0], stride=stem_config[i][1], bias=False)
            for l in [conv,]:
                nn.init.kaiming_uniform_(l.weight, a=1)
            layers.append(conv)

            # Initialize norm
            layers.append(group_norm(stem_channels[i]))

            # Initialize nonlinearity
            layers.append(nn.ReLU(inplace=True))

            # Update in_channels
            in_channels = stem_channels[i]


        # Combine layers into module
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)



################################################################################
### Stage
################################################################################
class Stage(nn.Module):
    """
    Like a ResNet stage, but more general
    Consists of a stack of Bottleneck modules
    """
    def __init__(
        self,
        config,
        channels,
        use_dcn=False,
        with_modulated_dcn=False,
        deformable_groups=1,
    ):
        super(Stage, self).__init__()
        blocks = []
        dcn_config = {
                "stage_with_dcn": use_dcn,
                "with_modulated_dcn": with_modulated_dcn,
                "deformable_groups": deformable_groups,
        }

        # Build each Bottleneck
        in_channels = channels[0]
        bottleneck_channels = channels[1]
        out_channels = channels[2]
        use_downsample = True
        block_count = len(config)
        for i in range(block_count):
            # if stride > 1, we need a downsample on the residual
            if config[i][1] > 1:
                use_downsample = True
            blocks.append(
                Bottleneck(
                    in_channels=in_channels,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel=config[i][0],
                    stride=config[i][1],
                    padding=config[i][2],
                    dilation=config[i][3],
                    use_downsample=use_downsample,
                    dcn_config=dcn_config,
                )
            )
            in_channels = out_channels
            use_downsample=False

        self.blocks = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.blocks(x)



################################################################################
### Bottleneck
################################################################################
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        kernel,
        stride,
        padding,
        dilation,
        use_downsample=False,
        num_groups=1,
        norm_func=group_norm,
        dcn_config={},
    ):
        super(Bottleneck, self).__init__()

        ### Downsample layer (on residual)
        # If use_downsample arg is set, need a downsample layer for the residual connections
        self.downsample = None
        if use_downsample:
            down_stride = stride #if dilation == 1 else 1  # Don't want this
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        ### First conv
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        

        ### Middle conv
        # Initialize padding layer
        pad_tuple = get_pad_tuple(padding)
        self.pad = nn.ZeroPad2d(pad_tuple)
        # Initialize conv layer
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=kernel,
                stride=stride,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            #padding = dilation
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel,
                stride=stride,
                #padding=padding,
                dilation=dilation,
                bias=False,
                groups=num_groups,
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)


        ### Third conv
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out



