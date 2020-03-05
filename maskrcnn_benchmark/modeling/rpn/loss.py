# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        #print("anchor:", anchor, anchor.bbox, anchor.fields())
        #print("target:", target, target.bbox, target.get_field('labels'))
        # Compute IoU between each target and anchor (tensor)
        match_quality_matrix = boxlist_iou(target, anchor)
        #print("match_quality_matrix:", match_quality_matrix.shape)
        # Use Matcher object to match every anchor to a GT (or none if negative)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        #print("matched_idxs:", matched_idxs, matched_idxs.shape)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        #print("matched_targets:", matched_targets, matched_targets.bbox)
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets, probe=False):
        #print("anchors:", anchors)
        #print("targets:", targets)
        labels = []
        regression_targets = []
        matched_targets_list = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if probe:
                matched_targets = self.match_targets_to_anchors(
                    anchors_per_image, targets_per_image, ['labels']
                )
                matched_targets_list.append(matched_targets)
            else:
                matched_targets = self.match_targets_to_anchors(
                    anchors_per_image, targets_per_image, self.copied_fields
                )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

            ### New: Isolate matched FG target boxes
            #matched_fg_idxs = labels_per_image.clamp(min=0).to(dtype=torch.int64)
            #print("matched_fg_idxs:", matched_fg_idxs, matched_fg_idxs.shape, matched_fg_idxs.min(), matched_fg_idxs.max(), matched_fg_idxs.sum())
            #matched_fg_targets = matched_targets.bbox[matched_fg_idxs.nonzero()].squeeze()
            #print("matched_fg_targets:", matched_fg_targets, matched_fg_targets.shape)
            #exit()

        if probe:
            return labels, regression_targets, matched_targets_list

        return labels, regression_targets


    def __call__(self, anchors, objectness, box_regression, targets, ignore_idxs=[]):
        """
        Arguments:
            anchors (list[list[BoxList]])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])
            ignore_idxs: list of batch idxs that we will ignore

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        # Merge anchors from all pyramid layers
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        #print("sampled_pos_inds:", sampled_pos_inds[0].shape, sampled_pos_inds[1].shape, sampled_pos_inds[0].sum(), sampled_pos_inds[1].sum())
        #print("sampled_neg_inds:", sampled_neg_inds[0].shape, sampled_neg_inds[1].shape, sampled_neg_inds[0].sum(), sampled_neg_inds[1].sum())
        
        # If we are ignoring ALL images in the batch, set final_scaler to 0 so we zero all losses
        if len(ignore_idxs) == len(sampled_pos_inds):
            final_scaler = 0

        # If we are ignoring none or some images in the batch, 
        # replace sampled_pos_inds[ignore_idxs] and sampled_neg_inds[ignore_idxs] with zero tensors
        # and set final_scaler to 1.
        else:
            final_scaler = 1
            for ignore_idx in ignore_idxs:
                sampled_pos_inds[ignore_idx] = torch.zeros_like(sampled_pos_inds[ignore_idx])
                sampled_neg_inds[ignore_idx] = torch.zeros_like(sampled_neg_inds[ignore_idx])

            #print("AFTER: sampled_pos_inds:", sampled_pos_inds[0].shape, sampled_pos_inds[1].shape, sampled_pos_inds[0].sum(), sampled_pos_inds[1].sum())
            #print("AFTER: sampled_neg_inds:", sampled_neg_inds[0].shape, sampled_neg_inds[1].shape, sampled_neg_inds[0].sum(), sampled_neg_inds[1].sum())
     

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss * final_scaler, box_loss * final_scaler


    ### Probe function, return what you want and play with it in tools/probe_loss.py script
    def probe(self, anchors, objectness, box_regression, targets):
        # Merge anchors from all pyramid layers
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        # Get labels, regression_targets and matched_targets_list 
        labels, regression_targets, matched_targets_list = self.prepare_targets(anchors, targets, probe=True)
        # Sample FG and BG anchor indices
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        #print("sampled_pos_inds:", sampled_pos_inds[0].shape, sampled_pos_inds[1].shape, sampled_pos_inds[0].sum(), sampled_pos_inds[1].sum())
        #print("sampled_neg_inds:", sampled_neg_inds[0].shape, sampled_neg_inds[1].shape, sampled_neg_inds[0].sum(), sampled_neg_inds[1].sum())
        #print("matched_targets.fields:", matched_targets_list[0].get_field('labels'))

        # Combine matched_targets boxes and labels over batch (same format as sampled_pos_inds ends up in)
        matched_target_boxes = torch.cat([bl.bbox for bl in matched_targets_list], dim=0)
        #print("matched_target_boxes:", matched_target_boxes, matched_target_boxes.shape)
        matched_target_classes = torch.cat([bl.get_field('labels') for bl in matched_targets_list], dim=0).unsqueeze(1)
        #print("matched_target_classes:", matched_target_classes, matched_target_classes.shape)
        # Concatenate boxes and classes
        matched_target_total = torch.cat((matched_target_boxes, matched_target_classes.to(dtype=torch.float32)), dim=1)
        #print("matched_target_total:", matched_target_total, matched_target_total.shape)

        #matched_fg_idxs = labels_per_image.clamp(min=0).to(dtype=torch.int64)
        #print("matched_fg_idxs:", matched_fg_idxs, matched_fg_idxs.shape, matched_fg_idxs.min(), matched_fg_idxs.max(), matched_fg_idxs.sum())
        #sampled_fg_target_boxes = matched_targets.bbox[matched_fg_idxs.nonzero()].squeeze()
        #print("matched_fg_targets:", matched_fg_targets, matched_fg_targets.shape)


        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        # Only take sampled_pos_inds
        matched_target_total = matched_target_total[sampled_pos_inds].squeeze()
        #print("matched_target_total:", matched_target_total, matched_target_total.shape)
        
        # We only care about sampled FG-matched anchors
        #sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        sampled_inds = sampled_pos_inds

        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        #print("labels:", labels, labels.shape)
        #print("regression_targets:", regression_targets, regression_targets.shape)

        # Compute objectness loss and add to total
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds], reduction='none'
        )
        #print("objectness_loss:", objectness_loss, objectness_loss.shape)
        # Concatenate objectness loss to matched_target_total
        matched_target_total = torch.cat((matched_target_total, objectness_loss.unsqueeze(1)), dim=1)
        #print("matched_target_total:", matched_target_total, matched_target_total.shape)

        # Compute bbox reg loss and add to total
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            #beta=1.0 / 9,
            reduction='none'
        )
        box_loss = torch.sum(box_loss, dim=1)
        #print("box_loss:", box_loss, box_loss.shape)

        # Concatenate bbox reg loss to matched_target_total
        matched_target_total = torch.cat((matched_target_total, box_loss.unsqueeze(1)), dim=1)
        #print("matched_target_total:", matched_target_total, matched_target_total.shape)
        
        return matched_target_total




# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
