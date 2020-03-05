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
### DDPP Backbone Modules
################################################################################
### DDPP with semi sharing
class DDPPv2_SS(nn.Module):
    """
    DDPP backbone module with semi sharing
    """
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs under cfg.MODEL.DDPP
        """
        super(DDPPv2_SS, self).__init__()
        # Assert correct config format
        assert (len(cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.DOWN_CHANNELS)), "Down block counts must equal down channels"
        assert (len(cfg.MODEL.DDPP.UP_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.UP_CHANNELS)), "Up block counts must equal up channels"

        # Set local option flags
        self.use_cascade_head = cfg.MODEL.DDPP.USE_CASCADE_HEAD
        self.use_cascade_body = cfg.MODEL.DDPP.USE_CASCADE_BODY
        self.use_hourglass_skip = cfg.MODEL.DDPP.USE_HOURGLASS_SKIP
        self.use_stem2x = cfg.MODEL.DDPP.USE_STEM2X

        # Construct Stem
        if self.use_stem2x:
            self.C1 = Stem2x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)
        else:
            self.C1 = Stem4x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)

        # Construct DownStages
        self.C2 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[0][0], cfg.MODEL.DDPP.DOWN_CHANNELS[0][1], cfg.MODEL.DDPP.DOWN_CHANNELS[0][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[0])

        self.C3_L1 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.DOWN_CHANNELS[1][1], cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[1])
        self.C3_L5 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.DOWN_CHANNELS[1][1], cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[1])

        self.C4_L1 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C4_L4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C4_L5 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        
        self.C5_L1 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])
        self.C5_L3 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])
        self.C5_L4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])
        self.C5_L5 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])

        # Construct UpStages
        self.D2 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[0][0], cfg.MODEL.DDPP.UP_CHANNELS[0][1], cfg.MODEL.DDPP.UP_CHANNELS[0][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[0])
        self.D3 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][1], cfg.MODEL.DDPP.UP_CHANNELS[1][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[1])
        self.D4 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][1], cfg.MODEL.DDPP.UP_CHANNELS[2][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[2])
        self.D5 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[3][0], cfg.MODEL.DDPP.UP_CHANNELS[3][1], cfg.MODEL.DDPP.UP_CHANNELS[3][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[3])

        # Construct ChannelReduction layers
        self.chred_L1 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L2 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L3 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L4 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L5 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)

        ### OPTIONAL LAYER CONSTRUCTION
        # Construct hourglass_skip layers
        #if self.use_hourglass_skip:
        #    self.hourglass_skip2 = Conv2d(cfg.MODEL.DDPP.STEM_OUT_CHANNELS, cfg.MODEL.DDPP.UP_CHANNELS[0][2], kernel_size=1, stride=1)
        #    self.hourglass_skip3 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][2], kernel_size=1, stride=1)
        #    self.hourglass_skip4 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][2], kernel_size=1, stride=1)

        # Construct post_cascade layers
        if self.use_cascade_head:
            self.post_cascade_head_blocks = []
            for i in range(1, 6):
                # Set block_name
                block_name = "post_cascade_head_block_L" + str(i)
                # Construct block_module
                block_module = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=3, stride=1, padding=1)
                # Add block to module
                self.add_module(block_name, block_module)
                # Add block name to list
                self.post_cascade_head_blocks.append(block_name)


    def forward(self, x):
        if self.use_cascade_body:
            return self.forward_cascade(x)
        #else:
        #    return self.forward_vanilla(x)

    def forward_cascade(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs (tuple[Tensor]): output feature maps from DDPP.
                They are ordered from highest resolution first (like FPN).
        """
        outputs = []
        # Forward thru stem
        C1_out = self.C1(x)

        # Level 5
        C2_out = self.C2(C1_out)
        C3_L5_out = self.C3_L5(C2_out)
        C4_L5_out = self.C4_L5(C3_L5_out)
        C5_L5_out = self.C5_L5(C4_L5_out)
        outputs.insert(0, self.chred_L5(C5_L5_out))

        # Level 4
        D2_out = self.D2(C2_out)
        D2_out_fused = D2_out + F.interpolate(C2_out, size=D2_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C3_L1_out = self.C3_L1(D2_out_fused)
        C3_L1_out_fused = C3_L1_out + F.interpolate(C3_L5_out, size=C3_L1_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C4_L4_out = self.C4_L4(C3_L1_out_fused)
        C4_L4_out_fused = C4_L4_out + F.interpolate(C4_L5_out, size=C4_L4_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C5_L4_out = self.C5_L4(C4_L4_out_fused)
        C5_L4_out_fused = C5_L4_out + F.interpolate(C5_L5_out, size=C5_L4_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L4(C5_L4_out_fused))

        # Level 3
        D3_out = self.D3(C3_L1_out)
        D3_out_fused = D3_out + F.interpolate(C3_L1_out_fused, size=D3_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C4_L1_out = self.C4_L1(D3_out_fused)
        C4_L1_out_fused = C4_L1_out + F.interpolate(C4_L4_out_fused, size=C4_L1_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C5_L3_out = self.C5_L3(C4_L1_out_fused)
        C5_L3_out_fused = C5_L3_out + F.interpolate(C5_L4_out_fused, size=C5_L3_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L3(C5_L3_out_fused))

        # Level 2
        D4_out = self.D4(C4_L1_out)
        D4_out_fused = D4_out + F.interpolate(C4_L1_out_fused, size=D4_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C5_L1_out = self.C5_L1(D4_out_fused)
        C5_L1_out_fused = C5_L1_out + F.interpolate(C5_L3_out_fused, size=C5_L1_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L2(C5_L1_out_fused))

        # Level 1
        D5_out = self.D5(C5_L1_out)
        D5_out_fused = D5_out + F.interpolate(C5_L1_out_fused, size=D5_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L1(D5_out_fused))


        if self.use_cascade_head:
            new_outputs = []
            # Store top feature map (low res)
            last_inner = outputs[-1]
            # Forward top feature map thru its post_cascade block and store to new_outputs
            new_outputs.append(getattr(self, self.post_cascade_head_blocks[-1])(last_inner))
            # Iterate over remaining feat maps in reverse (top-down)
            for feature, post_cascade_head_block in zip(outputs[:-1][::-1], self.post_cascade_head_blocks[:-1][::-1]):
                # Upsample current feature
                inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode='bilinear', align_corners=False)
                # Fuse features
                last_inner = feature + inner_top_down
                # Forward fused feature through post_cascade_head_block and insert to front of list
                new_outputs.insert(0, getattr(self, post_cascade_head_block)(last_inner))
            outputs = new_outputs
        
        return tuple(outputs)



### DDPP with full sharing
class DDPPv2(nn.Module):
    """
    DDPP backbone module with full sharing
    """
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs under cfg.MODEL.DDPP
        """
        super(DDPPv2, self).__init__()
        # Assert correct config format
        assert (len(cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.DOWN_CHANNELS)), "Down block counts must equal down channels"
        assert (len(cfg.MODEL.DDPP.UP_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.UP_CHANNELS)), "Up block counts must equal up channels"

        # Set local option flags
        self.use_cascade_head = cfg.MODEL.DDPP.USE_CASCADE_HEAD
        self.use_cascade_body = cfg.MODEL.DDPP.USE_CASCADE_BODY
        self.use_hourglass_skip = cfg.MODEL.DDPP.USE_HOURGLASS_SKIP
        self.use_stem2x = cfg.MODEL.DDPP.USE_STEM2X
        self.use_intermediate_supervision = cfg.MODEL.DDPP.USE_INTERMEDIATE_SUPERVISION

        # Construct Stem
        if self.use_stem2x:
            self.C1 = Stem2x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)
        else:
            self.C1 = Stem4x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)

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

        # Construct ChannelReduction layers
        self.chred_L1 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L2 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L3 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L4 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L5 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)

        ### OPTIONAL LAYER CONSTRUCTION
        # Construct hourglass_skip layers
        #if self.use_hourglass_skip:
        #    self.hourglass_skip2 = Conv2d(cfg.MODEL.DDPP.STEM_OUT_CHANNELS, cfg.MODEL.DDPP.UP_CHANNELS[0][2], kernel_size=1, stride=1)
        #    self.hourglass_skip3 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][2], kernel_size=1, stride=1)
        #    self.hourglass_skip4 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][2], kernel_size=1, stride=1)

        # Construct post_cascade layers
        if self.use_cascade_head:
            self.post_cascade_head_blocks = []
            for i in range(1, 6):
                # Set block_name
                block_name = "post_cascade_head_block_L" + str(i)
                # Construct block_module
                block_module = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=3, stride=1, padding=1)
                # Add block to module
                self.add_module(block_name, block_module)
                # Add block name to list
                self.post_cascade_head_blocks.append(block_name)


    def forward(self, x):
        if self.use_cascade_body:
            return self.forward_cascade(x)
        else:
            return self.forward_vanilla(x)

    def forward_cascade(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs (tuple[Tensor]): output feature maps from DDPP.
                They are ordered from highest resolution first (like FPN).
        """
        outputs = []
        # Forward thru stem
        C1_out = self.C1(x)

        # Level 5
        C2_out = self.C2(C1_out)
        C3_L5_out = self.C3(C2_out)
        C4_L5_out = self.C4(C3_L5_out)
        C5_L5_out = self.C5(C4_L5_out)
        outputs.insert(0, self.chred_L5(C5_L5_out))

        # Level 4
        D2_out = self.D2(C2_out)
        D2_out_fused = D2_out + F.interpolate(C2_out, size=D2_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C3_L1_out = self.C3(D2_out_fused)
        C3_L1_out_fused = C3_L1_out + F.interpolate(C3_L5_out, size=C3_L1_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C4_L4_out = self.C4(C3_L1_out_fused)
        C4_L4_out_fused = C4_L4_out + F.interpolate(C4_L5_out, size=C4_L4_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C5_L4_out = self.C5(C4_L4_out_fused)
        C5_L4_out_fused = C5_L4_out + F.interpolate(C5_L5_out, size=C5_L4_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L4(C5_L4_out_fused))

        # Level 3
        D3_out = self.D3(C3_L1_out)
        D3_out_fused = D3_out + F.interpolate(C3_L1_out_fused, size=D3_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C4_L1_out = self.C4(D3_out_fused)
        C4_L1_out_fused = C4_L1_out + F.interpolate(C4_L4_out_fused, size=C4_L1_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C5_L3_out = self.C5(C4_L1_out_fused)
        C5_L3_out_fused = C5_L3_out + F.interpolate(C5_L4_out_fused, size=C5_L3_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L3(C5_L3_out_fused))

        # Level 2
        D4_out = self.D4(C4_L1_out)
        D4_out_fused = D4_out + F.interpolate(C4_L1_out_fused, size=D4_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        C5_L1_out = self.C5(D4_out_fused)
        C5_L1_out_fused = C5_L1_out + F.interpolate(C5_L3_out_fused, size=C5_L1_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L2(C5_L1_out_fused))

        # Level 1
        D5_out = self.D5(C5_L1_out)
        D5_out_fused = D5_out + F.interpolate(C5_L1_out_fused, size=D5_out.shape[-2:], mode='bilinear', align_corners=False) / 2
        outputs.insert(0, self.chred_L1(D5_out_fused))


        if self.use_cascade_head:
            new_outputs = []
            # Store top feature map (low res)
            last_inner = outputs[-1]
            # Forward top feature map thru its post_cascade block and store to new_outputs
            new_outputs.append(getattr(self, self.post_cascade_head_blocks[-1])(last_inner))
            # Iterate over remaining feat maps in reverse (top-down)
            for feature, post_cascade_head_block in zip(outputs[:-1][::-1], self.post_cascade_head_blocks[:-1][::-1]):
                # Upsample current feature
                inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode='bilinear', align_corners=False)
                # Fuse features
                last_inner = feature + inner_top_down
                # Forward fused feature through post_cascade_head_block and insert to front of list
                new_outputs.insert(0, getattr(self, post_cascade_head_block)(last_inner))
            outputs = new_outputs
        
        if self.use_intermediate_supervision:
            intermediate_outputs = [D2_out_fused, D3_out_fused, D4_out_fused, D5_out_fused]
            return tuple(outputs), tuple(intermediate_outputs)
        else:
            return tuple(outputs)


    def forward_vanilla(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs (tuple[Tensor]): output feature maps from DDPP.
                They are ordered from highest resolution first (like FPN).
        """
        outputs = []
        # Forward thru stem
        C1_out = self.C1(x)

        # Level 1 (base)
        #if self.use_hourglass_skip:
        #    C2_out = self.C2(C1_out)
        #    D2_out = self.D2(C2_out)
        #    D2_out_fused = D2_out + self.hourglass_skip1(C1_out)
        #    C3_out = self.C3(D2_out_fused)
        #    D3_out = self.D3(C3_out)
        #    D3_out_fused = D3_out + self.hourglass_skip2(D2_out)
        #    C4_out = self.C4(D3_out_fused)
        #    D4_out = self.D4(C4_out)
        #    D4_out_fused = D4_out + self.hourglass_skip3(D3_out)
        #    x = self.C5(D4_out_fused)
        #    x = self.chred1(x)
        #    outputs.append(x)
        #else:

        C2_out = self.C2(C1_out)
        D2_out = self.D2(C2_out)
        C3_out = self.C3(D2_out)
        D3_out = self.D3(C3_out)
        C4_out = self.C4(D3_out)
        D4_out = self.D4(C4_out)
        C5_out = self.C5(D4_out)
        D5_out = self.D5(C5_out)
        x = self.chred_L1(D5_out)
        outputs.append(x)
            
        # Level 2
        x = self.chred_L2(C5_out)
        outputs.append(x)

        # Level 3
        x = self.C5(C4_out)
        x = self.chred_L3(x)
        outputs.append(x)

        # Level 4
        x = self.C4(C3_out)
        x = self.C5(x)
        x = self.chred_L4(x)
        outputs.append(x)

        # Level 5
        x = self.C3(C2_out)
        x = self.C4(x)
        x = self.C5(x)
        x = self.chred_L5(x)
        outputs.append(x)

        if self.use_cascade_head:
            new_outputs = []
            # Store top feature map (low res)
            last_inner = outputs[-1]
            # Forward top feature map thru its post_cascade block and store to new_outputs
            new_outputs.append(getattr(self, self.post_cascade_head_blocks[-1])(last_inner))
            # Iterate over remaining feat maps in reverse (top-down)
            for feature, post_cascade_head_block in zip(outputs[:-1][::-1], self.post_cascade_head_blocks[:-1][::-1]):
                # Upsample current feature
                inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode='bilinear', align_corners=False)
                # Fuse features
                last_inner = feature + inner_top_down
                # Forward fused feature through post_cascade_head_block and insert to front of list
                new_outputs.insert(0, getattr(self, post_cascade_head_block)(last_inner))
            outputs = new_outputs
        
        if self.use_intermediate_supervision:
            intermediate_outputs = [D2_out_fused, D3_out_fused, D4_out_fused, D5_out_fused]
            return tuple(outputs), tuple(intermediate_outputs)
        else:
            return tuple(outputs)



### DDPP with full sharing
class DDPP(nn.Module):
    """
    DDPP backbone module with full sharing
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

        # Set local option flags
        self.use_cascade = cfg.MODEL.DDPP.USE_CASCADE
        self.use_hourglass_skip = cfg.MODEL.DDPP.USE_HOURGLASS_SKIP
        self.use_stem2x = cfg.MODEL.DDPP.USE_STEM2X

        # Construct Stem
        if self.use_stem2x:
            self.C1 = Stem2x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)
        else:
            self.C1 = Stem4x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)

        # Construct DownStages
        self.C2 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[0][0], cfg.MODEL.DDPP.DOWN_CHANNELS[0][1], cfg.MODEL.DDPP.DOWN_CHANNELS[0][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[0])
        self.C3 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.DOWN_CHANNELS[1][1], cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[1])
        self.C4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C5 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])

        # Construct UpStages
        self.D2 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[0][0], cfg.MODEL.DDPP.UP_CHANNELS[0][1], cfg.MODEL.DDPP.UP_CHANNELS[0][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[0])
        self.D3 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][1], cfg.MODEL.DDPP.UP_CHANNELS[1][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[1])
        self.D4 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][1], cfg.MODEL.DDPP.UP_CHANNELS[2][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[2])

        # Construct ChannelReduction layers
        self.chred1 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred2 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred3 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred4 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)

        # Construct top_block
        self.top_block = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        ### OPTIONAL LAYER CONSTRUCTION
        # Construct hourglass_skip layers
        if self.use_hourglass_skip:
            self.hourglass_skip1 = Conv2d(cfg.MODEL.DDPP.STEM_OUT_CHANNELS, cfg.MODEL.DDPP.UP_CHANNELS[0][2], kernel_size=1, stride=1)
            self.hourglass_skip2 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][2], kernel_size=1, stride=1)
            self.hourglass_skip3 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][2], kernel_size=1, stride=1)

        # Construct post_cascade layers
        if self.use_cascade:
            self.post_cascade_blocks = []
            for i in range(1, 5):
                # Set block_name
                block_name = "post_cascade" + str(i)
                # Construct block_module
                block_module = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=3, stride=1, padding=1)
                # Add block to module
                self.add_module(block_name, block_module)
                # Add block name to list
                self.post_cascade_blocks.append(block_name)



    def forward(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs (tuple[Tensor]): output feature maps from DDPP.
                They are ordered from highest resolution first (like FPN).
        """
        outputs = []
        # Forward thru stem
        C1_out = self.C1(x)

        # Level 1 (base)
        if self.use_hourglass_skip:
            C2_out = self.C2(C1_out)
            D2_out = self.D2(C2_out)
            D2_out_fused = D2_out + self.hourglass_skip1(C1_out)
            C3_out = self.C3(D2_out_fused)
            D3_out = self.D3(C3_out)
            D3_out_fused = D3_out + self.hourglass_skip2(D2_out)
            C4_out = self.C4(D3_out_fused)
            D4_out = self.D4(C4_out)
            D4_out_fused = D4_out + self.hourglass_skip3(D3_out)
            x = self.C5(D4_out_fused)
            x = self.chred1(x)
            outputs.append(x)
        else:
            C2_out = self.C2(C1_out)
            D2_out = self.D2(C2_out)
            C3_out = self.C3(D2_out)
            D3_out = self.D3(C3_out)
            C4_out = self.C4(D3_out)
            D4_out = self.D4(C4_out)
            x = self.C5(D4_out)
            x = self.chred1(x)
            outputs.append(x)
            
        # Level 2
        x = self.C5(C4_out)
        x = self.chred2(x)
        outputs.append(x)

        # Level 3
        x = self.C4(C3_out)
        x = self.C5(x)
        x = self.chred3(x)
        outputs.append(x)

        # Level 4
        x = self.C3(C2_out)
        x = self.C4(x)
        x = self.C5(x)
        x = self.chred4(x)
        outputs.append(x)

        if self.use_cascade:
            new_outputs = []
            # Store top feature map (low res)
            last_inner = outputs[-1]
            # Forward top feature map thru its post_cascade block and store to new_outputs
            new_outputs.append(getattr(self, self.post_cascade_blocks[-1])(last_inner))
            # Iterate over remaining feat maps in reverse (top-down)
            for feature, post_cascade_block in zip(outputs[:-1][::-1], self.post_cascade_blocks[:-1][::-1]):
                # Upsample current feature
                inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode='bilinear', align_corners=False)
                # Fuse features
                last_inner = feature + inner_top_down
                # Forward fused feature through post_cascade_block and insert to front of list
                new_outputs.insert(0, getattr(self, post_cascade_block)(last_inner))
 
            outputs = new_outputs
            x = outputs[-1]

        # Level 5 (top)
        x = self.top_block(x)
        outputs.append(x)
        
        return tuple(outputs)



#################################################################
### DDPP with semi-sharing
class DDPP_SS(nn.Module):
    """
    DDPP_SS backbone module (semi-share)
    """
    def __init__(self, cfg):
        """
        Arguments:
            cfg object which contains necessary configs under cfg.MODEL.DDPP
        """
        super(DDPP_SS, self).__init__()
        # Assert correct config format
        assert (len(cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.DOWN_CHANNELS)), "Down block counts must equal down channels"
        assert (len(cfg.MODEL.DDPP.UP_BLOCK_COUNTS) == len(cfg.MODEL.DDPP.UP_CHANNELS)), "Up block counts must equal up channels"

        # Set local option flags
        self.use_cascade = cfg.MODEL.DDPP.USE_CASCADE
        self.use_hourglass_skip = cfg.MODEL.DDPP.USE_HOURGLASS_SKIP
        self.use_stem2x = cfg.MODEL.DDPP.USE_STEM2X

        # Construct Stem
        if self.use_stem2x:
            self.C1 = Stem2x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)
        else:
            self.C1 = Stem4x(cfg.MODEL.DDPP.STEM_OUT_CHANNELS)

        # Construct DownStages
        self.C2 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[0][0], cfg.MODEL.DDPP.DOWN_CHANNELS[0][1], cfg.MODEL.DDPP.DOWN_CHANNELS[0][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[0])
        self.C3_L1 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.DOWN_CHANNELS[1][1], cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[1])
        self.C3_L4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.DOWN_CHANNELS[1][1], cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[1])
        self.C4_L1 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C4_L3 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C4_L4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.DOWN_CHANNELS[2][1], cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[2])
        self.C5_L1 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])
        self.C5_L2 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])
        self.C5_L3 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])
        self.C5_L4 = DownStage(cfg.MODEL.DDPP.DOWN_CHANNELS[3][0], cfg.MODEL.DDPP.DOWN_CHANNELS[3][1], cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.DOWN_BLOCK_COUNTS[3])

        # Construct UpStages
        self.D2 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[0][0], cfg.MODEL.DDPP.UP_CHANNELS[0][1], cfg.MODEL.DDPP.UP_CHANNELS[0][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[0])
        self.D3 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][1], cfg.MODEL.DDPP.UP_CHANNELS[1][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[1])
        self.D4 = UpStage(cfg.MODEL.DDPP.UP_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][1], cfg.MODEL.DDPP.UP_CHANNELS[2][2], cfg.MODEL.DDPP.UP_BLOCK_COUNTS[2])

        # Construct ChannelReduction layers
        self.chred_L1 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L2 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L3 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)
        self.chred_L4 = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_BEFORE_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=1, stride=1)

        # Construct top_block
        self.top_block = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        ### OPTIONAL LAYER CONSTRUCTION
        # Construct hourglass_skip layers
        if self.use_hourglass_skip:
            self.hourglass_skip2 = Conv2d(cfg.MODEL.DDPP.STEM_OUT_CHANNELS, cfg.MODEL.DDPP.UP_CHANNELS[0][2], kernel_size=1, stride=1)
            self.hourglass_skip3 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[1][0], cfg.MODEL.DDPP.UP_CHANNELS[1][2], kernel_size=1, stride=1)
            self.hourglass_skip4 = Conv2d(cfg.MODEL.DDPP.DOWN_CHANNELS[2][0], cfg.MODEL.DDPP.UP_CHANNELS[2][2], kernel_size=1, stride=1)

        # Construct post_cascade layers
        if self.use_cascade:
            self.post_cascade_blocks = []
            for i in range(1, 5):
                # Set block_name
                block_name = "post_cascade" + str(i)
                # Construct block_module
                block_module = Conv2d(cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, cfg.MODEL.DDPP.OUT_CHANNELS_AFTER_CHRED, kernel_size=3, stride=1, padding=1)
                # Add block to module
                self.add_module(block_name, block_module)
                # Add block name to list
                self.post_cascade_blocks.append(block_name)



    def forward(self, x):
        """
        Arguments:
            x (Tensor): Input image batch
        Returns:
            outputs (tuple[Tensor]): output feature maps from DDPP.
                They are ordered from highest resolution first (like FPN).
        """
        outputs = []
        # Forward thru stem
        C1_out = self.C1(x)

        # Level 1 (base)
        if self.use_hourglass_skip:
            C2_out = self.C2(C1_out)
            D2_out = self.D2(C2_out)
            D2_out_fused = D2_out + self.hourglass_skip2(C1_out)
            C3_L1_out = self.C3_L1(D2_out_fused)
            D3_out = self.D3(C3_L1_out)
            D3_out_fused = D3_out + self.hourglass_skip3(D2_out)
            C4_L1_out = self.C4_L1(D3_out_fused)
            D4_out = self.D4(C4_L1_out)
            D4_out_fused = D4_out + self.hourglass_skip4(D3_out)
            x = self.C5_L1(D4_out_fused)
            x = self.chred_L1(x)
            outputs.append(x)
        else:
            C2_out = self.C2(C1_out)
            D2_out = self.D2(C2_out)
            C3_L1_out = self.C3_L1(D2_out)
            D3_out = self.D3(C3_L1_out)
            C4_L1_out = self.C4_L1(D3_out)
            D4_out = self.D4(C4_L1_out)
            x = self.C5_L1(D4_out)
            x = self.chred_L1(x)
            outputs.append(x)
            
        # Level 2
        x = self.C5_L2(C4_L1_out)
        x = self.chred_L2(x)
        outputs.append(x)

        # Level 3
        x = self.C4_L3(C3_L1_out)
        x = self.C5_L3(x)
        x = self.chred_L3(x)
        outputs.append(x)

        # Level 4
        x = self.C3_L4(C2_out)
        x = self.C4_L4(x)
        x = self.C5_L4(x)
        x = self.chred_L4(x)
        outputs.append(x)

        if self.use_cascade:
            new_outputs = []
            # Store top feature map (low res)
            last_inner = outputs[-1]
            # Forward top feature map thru its post_cascade block and store to new_outputs
            new_outputs.append(getattr(self, self.post_cascade_blocks[-1])(last_inner))
            # Iterate over remaining feat maps in reverse (top-down)
            for feature, post_cascade_block in zip(outputs[:-1][::-1], self.post_cascade_blocks[:-1][::-1]):
                # Upsample current feature
                inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode='bilinear', align_corners=False)
                # Fuse features
                last_inner = feature + inner_top_down
                # Forward fused feature through post_cascade_block and insert to front of list
                new_outputs.insert(0, getattr(self, post_cascade_block)(last_inner))
 
            outputs = new_outputs
            x = outputs[-1]

        # Level 5 (top)
        x = self.top_block(x)
        outputs.append(x)
        
        return tuple(outputs)



################################################################################
### DDPP Stem
################################################################################
class Stem2x(nn.Module):
    """
    Stem2x module
    2x downsample, group norm
    """
    def __init__(self, out_channels):
        super(Stem2x, self).__init__()
    
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


class Stem4x(nn.Module):
    """
    Stem4x module
    4x downsample, group norm
    """
    def __init__(self, out_channels):
        super(Stem4x, self).__init__()
    
        intermediate_channels = out_channels

        # 2x downsample
        self.conv1 = Conv2d(
            3, intermediate_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = group_norm(intermediate_channels)

        # 2x downsample
        self.conv2 = Conv2d(
            intermediate_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = group_norm(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)
        for l in [self.conv2,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
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
        in_channels,
        bottleneck_channels,
        out_channels,
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
                    in_channels,
                    bottleneck_channels,
                    out_channels,
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


