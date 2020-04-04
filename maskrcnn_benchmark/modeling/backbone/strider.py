# Matthew Inkawhich

"""
Define the Strider backbone module.
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
### Strider Backbone Module
################################################################################
class Strider(nn.Module):
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs under cfg.MODEL.STRIDER
        """
        super(Strider, self).__init__()
        # Assert correct config format
        assert (len(cfg.MODEL.STRIDER.STEM_CONFIG) == len(cfg.MODEL.STRIDER.STEM_CHANNELS)), "Stem config must equal stem channels"
        assert (len(cfg.MODEL.STRIDER.BODY_CHANNELS) == len(cfg.MODEL.STRIDER.BODY_CONFIGS)), "Body channels config must equal body configs"
        assert (len(cfg.MODEL.STRIDER.BODY_CHANNELS) == len(cfg.MODEL.STRIDER.STRIDERBLOCK_FUSEINTO_INDEXES)), "Body channels config must equal striderblock fuseinto indexes"
        assert (len(cfg.MODEL.STRIDER.BODY_CHANNELS) == len(cfg.MODEL.STRIDER.RETURN_FEATURES)), "Body channels config must equal return features"

        # Construct Stem
        self.stem = Stem(cfg.MODEL.STRIDER.STEM_CONFIG, cfg.MODEL.STRIDER.STEM_CHANNELS)

        # Construct Blocks
        self.block_names = []
        self.return_features = {}
        self.fuseinto_indexes = cfg.MODEL.STRIDER.STRIDERBLOCK_FUSEINTO_INDEXES
        body_channels = cfg.MODEL.STRIDER.BODY_CHANNELS
        body_configs = cfg.MODEL.STRIDER.BODY_CONFIGS
        return_features = cfg.MODEL.STRIDER.RETURN_FEATURES
        for i in range(len(body_channels)):
            name = "block" + str(i)
            in_channels = body_channels[i][0]
            bottleneck_channels = body_channels[i][1]
            out_channels = body_channels[i][2]

            # If the current element of reg_bottlenecks is not empty, build a regular Bottleneck
            if body_configs[i][0] == 0:
                stride = body_configs[i][1][0]
                dilation = body_configs[i][1][1]
                block = Bottleneck(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                            stride=stride,
                            dilation=dilation,
                        )
            
            # Else, build a StriderBlock
            else:
                block = StriderBlock(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                        )

            self.add_module(name, block)
            self.block_names.append(name)
            self.return_features[name] = return_features[i]
                


    def forward(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs ([Tensor]): Final output feature maps
        """
        #print("input:", x.shape)
        outputs = []
        x = self.stem(x)
        #print("stem:", x.shape)
        for i, block_name in enumerate(self.block_names):
            x = getattr(self, block_name)(x, self.fuseinto_indexes[i])
            #print(i, block_name, x.shape)
            if self.return_features[block_name]:
                #print("Adding to return list")
                outputs.append(x)
        
        #for i in range(len(outputs)):
        #    print("\ni:", i)
        #    print("shape:", outputs[i].shape)
        #    print("mean activation:", outputs[i].mean())
        #    print("frac of nonzero activations:", (outputs[i] != 0).sum().float() / outputs[i].numel())
        #exit()

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
### StriderBlock
################################################################################
class StriderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        norm_func=group_norm,
    ):
        super(StriderBlock, self).__init__()
        
        self.num_groups = num_groups

        ### Residual layer
        self.residual = nn.Sequential(
            Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=1, bias=False
            ),
            norm_func(out_channels),
        )
        for modules in [self.residual,]:
            for l in modules.modules():
                if isinstance(l, Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

        ### Conv1
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        

        ### Conv2
        # Conv2 must be represented by a tensor instead of a Module, as we
        # need to use the weights with different strides.
        self.conv2_weight = nn.Parameter(
            torch.Tensor(bottleneck_channels, bottleneck_channels // num_groups, 3, 3)
        )
        nn.init.kaiming_uniform_(self.conv2_weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)


        ### Conv3
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)



    def forward(self, x, fuseinto):
        # Store copy of input feature
        identity = x

        # Forward thru residual conv
        residual_out = self.residual(identity)

        # Forward thru conv1
        out = self.conv1(x)
        out = self.bn1(out)
        conv1_out = F.relu_(out)
    
        # Forward thru all branches
        # 2x DOWN branch
        branch_outputs = []
        # Conv2
        out = F.conv2d(conv1_out, self.conv2_weight, stride=2, padding=1, groups=self.num_groups)
        out = self.bn2(out)
        out = F.relu_(out)
        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)
        # Add residual
        out += F.avg_pool2d(residual_out, kernel_size=3, stride=2, padding=1)
        out = F.relu_(out)
        branch_outputs.append(out)

        # 1x SAME branch
        # Conv2
        out = F.conv2d(conv1_out, self.conv2_weight, stride=1, padding=1, groups=self.num_groups)
        out = self.bn2(out)
        out = F.relu_(out)
        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)
        # Add residual
        out += residual_out
        out = F.relu_(out)
        branch_outputs.append(out)

        # 2x UP branch
        # (T)Conv2
        out = F.conv_transpose2d(conv1_out, self.conv2_weight, stride=2, padding=1, output_padding=1, groups=self.num_groups)
        out = self.bn2(out)
        out = F.relu_(out)
        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)
        # Add residual
        out += F.interpolate(residual_out, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = F.relu_(out)
        branch_outputs.append(out)

        #print("\n")
        #for i in range(len(branch_outputs)):
        #    print(i, branch_outputs[i].shape)

        # Fuse features
        if fuseinto == 0:
            out = (branch_outputs[1] + F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1))
            out = (branch_outputs[0] + F.avg_pool2d(out, kernel_size=3, stride=2, padding=1))
        elif fuseinto == 1:
            out = (branch_outputs[1] + F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='bilinear', align_corners=False))
            out = (out + F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1))
        elif fuseinto == 2:
            out = (branch_outputs[1] + F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='bilinear', align_corners=False))
            out = (branch_outputs[2] + F.interpolate(out, size=branch_outputs[2].shape[-2:], mode='bilinear', align_corners=False))
        else:
            print("Error: Invalid fuseinto parameter in StriderBlock forward function")
            exit()
        
        #out = F.relu_(out)
        return out / 1.5




################################################################################
### Bottleneck
################################################################################
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride,
        dilation,
        use_downsample=False,
        num_groups=1,
        norm_func=group_norm,
    ):
        super(Bottleneck, self).__init__()

        ### Downsample layer (on residual)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            down_stride = stride
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
        padding = dilation
        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
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


    def forward(self, x, dummy):
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



