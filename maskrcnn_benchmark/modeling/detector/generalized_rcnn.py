# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.using_intermediate_supervision = False
    
        # Construct intermediate RPNs if necessary
        if cfg.MODEL.RPN.USE_DDPP and cfg.MODEL.DDPP.USE_INTERMEDIATE_SUPERVISION:
            self.using_intermediate_supervision = True
            self.irpn2 = build_rpn(cfg, cfg.MODEL.DDPP.DOWN_CHANNELS[0][2], cfg.MODEL.DDPP.IRPN_CONFIG[0])
            self.irpn3 = build_rpn(cfg, cfg.MODEL.DDPP.DOWN_CHANNELS[1][2], cfg.MODEL.DDPP.IRPN_CONFIG[1])
            self.irpn4 = build_rpn(cfg, cfg.MODEL.DDPP.DOWN_CHANNELS[2][2], cfg.MODEL.DDPP.IRPN_CONFIG[2])
            self.irpn5 = build_rpn(cfg, cfg.MODEL.DDPP.DOWN_CHANNELS[3][2], cfg.MODEL.DDPP.IRPN_CONFIG[3])
            self.irpn_loss_weight = cfg.MODEL.DDPP.IRPN_LOSS_WEIGHT

        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)


    def forward(self, images, targets=None, probe=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        #print("images:", images.tensors.size())
        if self.using_intermediate_supervision:
            # Forward thru backbone
            features, intermediate_features = self.backbone(images.tensors)

            if self.training:
                # Forward thru each irpn
                _, irpn2_losses = self.irpn2(images, [intermediate_features[0]], targets)
                _, irpn3_losses = self.irpn3(images, [intermediate_features[1]], targets)
                _, irpn4_losses = self.irpn4(images, [intermediate_features[2]], targets)
                _, irpn5_losses = self.irpn5(images, [intermediate_features[3]], targets)
                
        else:
            features = self.backbone(images.tensors)

        #print("features:", features[0].shape)
        #for idx, f in enumerate(features):
        #    print("feature[{}]:".format(idx), f.size())
        #exit()

        if probe:
            return self.rpn(images, features, targets, probe=True)

        proposals, proposal_losses = self.rpn(images, features, targets)
        #print("proposals:", proposals)
        #for idx, p in enumerate(proposals):
        #    print("proposal[{}]:".format(idx), p)
        #print("proposal_losses:", proposal_losses)
        #exit()

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        #print("roi_head.x:", x.size())
        #print("roi_head.result:", len(result))
        #for idx, r in enumerate(result):
        #    print("result[{}]:".format(idx), r, r.get_field('labels'))
        #print("detector_losses:", detector_losses)
        #exit()


        #print("result:", result)

        if self.training:
            losses = {}
            if self.using_intermediate_supervision:
                for k, v in irpn2_losses.items():
                    losses["irpn2_"+k] = v * self.irpn_loss_weight
                for k, v in irpn3_losses.items():
                    losses["irpn3_"+k] = v * self.irpn_loss_weight
                for k, v in irpn4_losses.items():
                    losses["irpn4_"+k] = v * self.irpn_loss_weight
                for k, v in irpn5_losses.items():
                    losses["irpn5_"+k] = v * self.irpn_loss_weight
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
