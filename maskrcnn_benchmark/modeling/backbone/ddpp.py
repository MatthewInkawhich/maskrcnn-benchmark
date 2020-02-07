# Matthew Inkawhich

"""
Defines all modules relating to DDPP backbone along
with the DDPP class itself.
"""
from collections import namedtuple

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
### DDPP Backbone Module
################################################################################
class DDPP(nn.Module):
    """
    DDPP backbone module
    """
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs under cfg.MODEL.DDPP
        """
        super(DDPP, self).__init__()
        # Assert correct config format
        assert (len(cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.DOWN_CHANNELS)), "Down block counts must equal down channels"
        assert (len(cfg.MODEL.DDPP.UP_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.UP_CHANNELS)), "Up block counts must equal up channels"

        # Construct Stem
        self.C1 = Stem(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)

        # Construct DownStages
        self.C2 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[0][0], cfg.MODEL.DDPP.DOWN_CHANNELS[0][1], cfg.MODEL.DDPP.DOWN_CHANNELS[0][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[0])
        self.C3 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.DOWN_CHANNELS[1][1], cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[1])
        self.C4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C5 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])

        # Construct UpStages
        self.D2 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[0][0], cfg.MODEL.DDPP.UP_CHANNELS[0][1], cfg.MODEL.DDPP.UP_CHANNELS[0][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[0])
        self.D3 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][1], cfg.MODEL.DDPP.UP_CHANNELS[1][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[1])
        self.D4 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][1], cfg.MODEL.DDPP.UP_CHANNELS[2][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[2])
        self.D5 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[3][0], cfg.MODEL.DDPP.UP_CHANNELS[3][1], cfg.MODEL.DDPP.UP_CHANNELS[3][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[3])


    def forward(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            results (tuple[Tensor]): output feature maps from DDPP.
                They are ordered from highest resolution first (like FPN).
        """

        return x


################################################################################
### DDPP Stem
################################################################################
class Stem(nn.Module):
    """
    Stem module
    2x downsample, group norm
    """
    def __init__(self, out_channels):
        super(Stem, self).__init__()
    
        # 2x downsample
        self.conv1 = Conv2d(
            3, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = group_norm(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        return x



################################################################################
### DDPP DownStage (conv)
################################################################################
class DownStage(nn.Module):
    """
    DownStage module
    Like a ResNet stage, but more general
    Consists of a stack of DownBottleneck modules
    """
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        block_count,
        first_stride=2,
        dilation=1,
        use_dcn=False,
        with_modulated_dcn=False,
        deformable_groups=1,
    ):
        super(DownStage, self).__init__()
        blocks = []
        stride = first_stride
        dcn_config = {
                "stage_with_dcn": use_dcn,
                "with_modulated_dcn": with_modulated_dcn,
                "deformable_groups": deformable_groups,
        }

        # Build each DownBottleneck
        for _ in range(block_count):
            blocks.append(
                DownBottleneck(
                    in_channels,
                    bottleneck_channels,
                    out_channels,
                    num_groups=1,
                    stride_in_1x1=False,
                    stride=stride,
                    dilation=dilation,
                    dcn_config=dcn_config,
                )
            )
            stride = 1
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.blocks(x)



################################################################################
### DDPP UpStage (tconv)
################################################################################
class UpStage(nn.Module):
    """
    UpStage module
    """
    def __init__(
        self,
        inout_channels,
        bottleneck_channels,
        block_count,
        last_stride=2,
    ):
        super(UpStage, self).__init__()

        blocks = []
        stride = 1

        # Build each UpBottleneck
        for i in range(block_count):
            # Upsample on last block
            if i == block_count-1:
                stride=last_stride
            blocks.append(
                UpBottleneck(
                    inout_channels,
                    bottleneck_channels,
                    inout_channels,
                    stride=stride,
                    dilation=1,
                    norm_func=group_norm,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.blocks(x)



################################################################################
### DDPP DownBottleneck (conv)
################################################################################
class DownBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=False,
        stride=1,
        dilation=1,
        norm_func=group_norm,
        dcn_config={},
    ):
        super(DownBottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
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

        # Commented this out... don't want this
        #if dilation > 1:
        #    stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )

        self.bn1 = norm_func(bottleneck_channels)
        
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            padding = dilation
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=padding,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )

        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

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



################################################################################
### DDPP UpBottleneck (tconv)
################################################################################
class UpBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride=1,
        dilation=1,
        norm_func=group_norm,
    ):
        super(UpBottleneck, self).__init__()

        # If the stride > 1, add upsample layer for residual connection
        self.upsample = None
        if stride > 1:
            self.upsample = nn.Sequential(
                ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False
                ),
                norm_func(out_channels),
            )
            #for modules in [self.downsample,]:
            #    for l in modules.modules():
            #        if isinstance(l, Conv2d):
            #            nn.init.kaiming_uniform_(l.weight, a=1)


        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.bn1 = norm_func(bottleneck_channels)
        
        # Add tconv layer
        if stride > 1:
            output_padding = 1
        else:
            output_padding = 0
        self.tconv = ConvTranspose2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=output_padding,
            bias=False,
            dilation=dilation
        )
        #nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels, 
            kernel_size=1, 
            bias=False
        )

        self.bn3 = norm_func(out_channels)

        #for l in [self.conv1, self.conv3,]:
        #    nn.init.kaiming_uniform_(l.weight, a=1)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.tconv(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = F.relu_(out)

        return out


