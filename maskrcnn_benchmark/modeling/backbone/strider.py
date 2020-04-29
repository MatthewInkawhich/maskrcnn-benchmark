# Matthew Inkawhich

"""
Define the Strider backbone module.
"""
from collections import namedtuple
import math
import random

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
        assert (len(cfg.MODEL.STRIDER.BODY_CHANNELS) == len(cfg.MODEL.STRIDER.BODY_CONFIG)), "Body channels config must equal body config"
        assert (len(cfg.MODEL.STRIDER.BODY_CHANNELS) == len(cfg.MODEL.STRIDER.OUTPUT_SIZES)), "Body channels config must equal output_sizes"
        assert (len(cfg.MODEL.STRIDER.BODY_CHANNELS) == len(cfg.MODEL.STRIDER.RETURN_FEATURES)), "Body channels config must equal return features"

        # Set norm func
        if cfg.MODEL.STRIDER.USE_GN:
            self.norm_func = group_norm
        else:
            self.norm_func = FrozenBatchNorm2d

        # Construct Stem
        if cfg.MODEL.STRIDER.STEM_CONFIG == "BASE":
            self.stem = BaseStem(cfg.MODEL.STRIDER.STEM_CHANNELS[0], self.norm_func)
        else:
            self.stem = Stem(cfg.MODEL.STRIDER.STEM_CONFIG, cfg.MODEL.STRIDER.STEM_CHANNELS, self.norm_func)

        # Construct Blocks
        self.block_names = []
        self.return_features = {}
        self.output_sizes = cfg.MODEL.STRIDER.OUTPUT_SIZES
        body_channels = cfg.MODEL.STRIDER.BODY_CHANNELS
        body_config = cfg.MODEL.STRIDER.BODY_CONFIG
        return_features = cfg.MODEL.STRIDER.RETURN_FEATURES
        stride_option = cfg.MODEL.STRIDER.STRIDE_OPTION
        full_residual = cfg.MODEL.STRIDER.FULL_RESIDUAL
        dilations = cfg.MODEL.STRIDER.DILATIONS
        weighted_fusion = cfg.MODEL.STRIDER.WEIGHTED_FUSION
        for i in range(len(body_channels)):
            name = "block" + str(i)
            in_channels = body_channels[i][0]
            bottleneck_channels = body_channels[i][1]
            out_channels = body_channels[i][2]

            # If the current element of reg_bottlenecks is not empty, build a regular Bottleneck
            if body_config[i][0] == 0:
                stride = body_config[i][1][0]
                dilation = body_config[i][1][1]
                block = Bottleneck(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                            stride=stride,
                            dilation=dilation,
                            norm_func=self.norm_func,
                            full_residual=full_residual,
                        )
            
            # Else, build a StriderBlock
            else:
                block = StriderBlock(
                            in_channels=in_channels,
                            bottleneck_channels=bottleneck_channels,
                            out_channels=out_channels,
                            stride_option=stride_option,
                            dilations=dilations,
                            norm_func=self.norm_func,
                            full_residual=full_residual,
                            weighted_fusion=weighted_fusion,
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
            print(i, block_name, x.shape)
            x = getattr(self, block_name)(x, self.output_sizes[i])
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
    def __init__(self, stem_config, stem_channels, norm_func):
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
            layers.append(norm_func(stem_channels[i]))
            # Initialize nonlinearity
            layers.append(nn.ReLU(inplace=True))
            # Update in_channels
            in_channels = stem_channels[i]

        # Combine layers into module
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)



class BaseStem(nn.Module):
    def __init__(self, out_channels, norm_func):
        super(BaseStem, self).__init__()
        # Define conv layer
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_func(out_channels)

        # Initialize conv
        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x



################################################################################
### WeightedFusionModule
################################################################################
class WeightedFusionModule(nn.Module):
    def __init__(self, in_channels, num_branches):
        super(WeightedFusionModule, self).__init__()
        self.conv = Conv2d(in_channels, num_branches, kernel_size=3, stride=1, padding=1)
        for l in [self.conv,]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x, output_size):
        # Forward thru conv
        x = self.conv(x)
        # Normalize channel values to percentages
        x = F.softmax(x, dim=1)
        # Resize output
        if output_size == 0:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        elif output_size == 2:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        # Depth-wise normalize
        s = x.sum(dim=1, keepdim=True)
        x = x / s
        # Convert to a list with each element representing a channel (i.e. branch)
        x = x.permute(1, 0, 2, 3).unsqueeze_(2)
        out = [a for a in x]
        return out
        



################################################################################
### StriderBlock
################################################################################
class StriderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride_option,
        dilations,
        norm_func=group_norm,
        full_residual=False,
        weighted_fusion=False,
    ):
        super(StriderBlock, self).__init__()
        
        self.stride_option = stride_option
        self.dilations = dilations
        self.weighted_fusion = weighted_fusion

        ### Initialize Weighted Fusion Module if necessary
        if self.weighted_fusion:
            self.wfm = WeightedFusionModule(in_channels, num_branches=3)

        ### Residual layer
        self.downsample = None
        if in_channels != out_channels or full_residual:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=1, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
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
            torch.Tensor(bottleneck_channels, bottleneck_channels, 3, 3)
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



    def forward(self, x, output_size):
        # Store copy of input feature
        identity = x

        # Forward thru WeightedFusionModule if necessary
        if self.weighted_fusion:
            fusion_weights = self.wfm(identity, output_size)

        # Forward thru residual conv
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Forward thru conv1
        out = self.conv1(x)
        out = self.bn1(out)
        conv1_out = F.relu_(out)
    
        # Forward thru all branches
        branch_outputs = []

        # 2x DOWN branch
        if self.stride_option in [0, 1, 2]:
            dilation = self.dilations[0]
            # Conv2
            out = F.conv2d(conv1_out, self.conv2_weight, stride=2, padding=dilation, dilation=dilation)
            out = self.bn2(out)
            out = F.relu_(out)
            # Conv3
            out = self.conv3(out)
            out = self.bn3(out)
            # Add residual
            out += F.avg_pool2d(identity, kernel_size=3, stride=2, padding=1)
            out = F.relu_(out)
            branch_outputs.append(out)

        # 1x SAME branch
        if self.stride_option in [0, 1, 3]:
            dilation = self.dilations[1]
            # Conv2
            out = F.conv2d(conv1_out, self.conv2_weight, stride=1, padding=dilation, dilation=dilation)
            out = self.bn2(out)
            out = F.relu_(out)
            # Conv3
            out = self.conv3(out)
            out = self.bn3(out)
            # Add residual
            out += identity
            out = F.relu_(out)
            branch_outputs.append(out)

        # 2x UP branch
        if self.stride_option in [0, 2, 3]:
            dilation = self.dilations[2]
            # (T)Conv2
            # Note: We want the Transposed conv with stride=2 to act like a conv with stride=1/2, 
            #       so we have to permute the in/out channels and flip the kernels to match the implementation.
            out = F.conv_transpose2d(conv1_out, self.conv2_weight.flip([2, 3]).permute(1, 0, 2, 3), stride=2, padding=dilation, output_padding=1, dilation=dilation)
            out = self.bn2(out)
            out = F.relu_(out)
            # Conv3
            out = self.conv3(out)
            out = self.bn3(out)
            # Add residual
            out += F.interpolate(identity, size=out.shape[-2:], mode='bilinear', align_corners=False)
            out = F.relu_(out)
            branch_outputs.append(out)

        
        # Resize branch outputs
        if output_size == 0:
            branch_outputs[1] = F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)
            branch_outputs[2] = F.avg_pool2d(F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1), kernel_size=3, stride=2, padding=1)
        elif output_size == 1:
            branch_outputs[0] = F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='bilinear', align_corners=False)
            branch_outputs[2] = F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1)
        elif output_size == 2:
            branch_outputs[0] = F.interpolate(branch_outputs[0], size=branch_outputs[2].shape[-2:], mode='bilinear', align_corners=False)
            branch_outputs[1] = F.interpolate(branch_outputs[1], size=branch_outputs[2].shape[-2:], mode='bilinear', align_corners=False)
        else:
            print("Error: Invalid output_size parameter in StriderBlock forward function")
            exit()

        #print("\n\n")
        #for fw in fusion_weights:
        #    print(fw, fw.shape)

        # Scale each branch output by weights (optional)
        if self.weighted_fusion:
            branch_outputs[0] = branch_outputs[0] * fusion_weights[0]
            branch_outputs[1] = branch_outputs[1] * fusion_weights[1]
            branch_outputs[2] = branch_outputs[2] * fusion_weights[2]

        # Fuse branch outputs
        out = branch_outputs[0] + branch_outputs[1] + branch_outputs[2]
        if not self.weighted_fusion:
            out = out / 3

        return out



#        if self.stride_option == 0:
#            if output_size == 0:
#                out = branch_outputs[1] + F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1)
#                out = (branch_outputs[0] + F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)) / 3
#            elif output_size == 1:
#                out = branch_outputs[1] + F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='bilinear', align_corners=False)
#                out = (out + F.avg_pool2d(branch_outputs[2], kernel_size=3, stride=2, padding=1)) / 3
#            elif output_size == 2:
#                out = branch_outputs[1] + F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='bilinear', align_corners=False)
#                out = (branch_outputs[2] + F.interpolate(out, size=branch_outputs[2].shape[-2:], mode='bilinear', align_corners=False)) / 3
#            else:
#                print("Error: Invalid output_size parameter in StriderBlock forward function")
#                exit()
#        elif self.stride_option == 1:
#            if output_size == 0:
#                out = (branch_outputs[0] + F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)) / 2
#            elif output_size == 1:
#                out = (branch_outputs[1] + F.interpolate(branch_outputs[0], size=branch_outputs[1].shape[-2:], mode='bilinear', align_corners=False)) / 2
#            else:
#                print("Error: Invalid output_size parameter in StriderBlock forward function")
#                exit()
#
#        elif self.stride_option == 2:
#            if output_size == 0:
#                out = F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)
#                out = (branch_outputs[0] + F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)) / 2
#            elif output_size == 1:
#                out = F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)
#                out = (out + F.interpolate(branch_outputs[0], size=out.shape[-2:], mode='bilinear', align_corners=False)) / 2
#            else:
#                print("Error: Invalid output_size parameter in StriderBlock forward function")
#                exit()
#
#        elif self.stride_option == 3:
#            if output_size == 0:
#                out = F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)
#                out = (F.avg_pool2d(branch_outputs[0], kernel_size=3, stride=2, padding=1) + F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)) / 2
#            elif output_size == 1:
#                out = (branch_outputs[0] + F.avg_pool2d(branch_outputs[1], kernel_size=3, stride=2, padding=1)) / 2
#            else:
#                print("Error: Invalid output_size parameter in StriderBlock forward function")
#                exit()

#        return out




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
        full_residual=False,
    ):
        super(Bottleneck, self).__init__()

        ### Downsample layer (on residual)
        self.downsample = None
        if in_channels != out_channels or full_residual:
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



