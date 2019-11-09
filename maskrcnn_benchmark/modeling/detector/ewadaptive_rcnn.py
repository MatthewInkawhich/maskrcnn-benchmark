# MatthewInkawhich

# This class contains and defines the element-wise adaptive r-cnn model

import torch
from torch import nn
import random

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone, build_resnet_stem, build_resnet_stage, build_ewa_selector
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class EWAdaptiveRCNN(nn.Module):

    def __init__(self, cfg):
        super(EWAdaptiveRCNN, self).__init__()

        self.cfg = cfg.clone()
        
        # Build ResNet stages
        self.C1 = build_resnet_stem(cfg)
        self.C2, C2_out_channels = build_resnet_stage(cfg, stage=2, in_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS)
        self.C3, C3_out_channels = build_resnet_stage(cfg, stage=3, in_channels=C2_out_channels)
        self.C4, C4_out_channels = build_resnet_stage(cfg, stage=4, in_channels=C3_out_channels)

        # Build selectors
        ewa_stage_specs = [cfg.MODEL.EWADAPTIVE.C2, cfg.MODEL.EWADAPTIVE.C3, cfg.MODEL.EWADAPTIVE.C4]
        out_channels = [C2_out_channels, C3_out_channels, C4_out_channels]
        self.selector_list = []
        for stage_idx in range(len(ewa_stage_specs)):
            num_branches = len(ewa_stage_specs[stage_idx])
            if num_branches > 1:
                selector = build_ewa_selector(out_channels[stage_idx], num_branches)
                name = "selector" + str(stage_idx + 2)
                self.add_module(name, selector)
                self.selector_list.append(name)

        # Build RPN
        self.rpn = build_rpn(cfg, C4_out_channels)
        # Build ROI heads
        self.roi_heads = build_roi_heads(cfg, C4_out_channels)
        # Declare supported forward options
        self.forward_options = ["inference",
                                "pretrain"]
        # Initialize route (elements correspond to the route taken thru each adaptive stage)
        self.route = (0, 0, 0)
        # Initialize synced flag, this represents whether or not the weights are synced
        self.synced = True


    def _get_random_branch_choice(self, num_branches):
        """
        Helper function to generate random branch index choice for pretraining
        """
        branch_choice = 0
        if num_branches > 1:
            branch_choice = random.randint(0, num_branches - 1)
        return branch_choice



    def sync_weights(self):
        """
        Syncs weights across branches for all submodules according to self.route tuple.
        """
        # Handle C2
        if len(self.C2.branches) > 1:
            # Multiple branches, must sync according to route
            for branch_name in self.C2.branches:
                updated_branch_name = "branch" + str(self.route[0])
                # Replace all branches that are not the updated_branch weights with the updated_branch's weights
                if branch_name != updated_branch_name:

            



    def forward(self, images, targets=None, option="inference"):
        # The 'option' arg defines what sub _forward we want to use
        if option not in self.forward_options:
            raise ValueError("Error: option arg is NOT valid! Must choose one of self.forward_options")
        if self.training and targets is None:
            raise ValueError("Error: In training mode, targets should be passed")
        
        # Choose forwarding option
        if option == "inference":
            return self._forward_inference(images)
        if option == "pretrain":
            return self._forward_pretrain(images, targets)


    def _forward_inference(self, images):
        """
        This function is used for inference, when we want our trained
        selectors to guide the adaptive processing.
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)

        ### Process image data with stages
        features_c1 = self.C1(images.tensors)
        features_c2 = self.C2(features_c1, branch=0)
        features_c3 = self.C3(features_c2, branch=0)
        features = [self.C4(features_c3, branch=0)] 

        # Forward thru RPN
        proposals, proposal_losses = self.rpn(images, features, targets=None)

        # Forward thru RoI heads
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets=None)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        return result


    def _forward_pretrain(self, images, targets):
        """
        This function is used for pretraining of the adaptive stages.
        It returns a losses dictionary which contains separate entries for
        each loss component.
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)
        #print("images:", images.tensors.size())


        ### Process image data with stages
        # Stem never has branches
        features_c1 = self.C1(images.tensors)
        
        # Choose random C2 branch
        branch_choice_c2 = self._get_random_branch_choice(len(self.C2.branches))
        print("branch choice c2:", branch_choice_c2)
        features_c2 = self.C2(features_c1, branch=branch_choice_c2)

        # Choose random C3 branch
        branch_choice_c3 = self._get_random_branch_choice(len(self.C3.branches))
        print("branch choice c3:", branch_choice_c3)
        features_c3 = self.C3(features_c2, branch=branch_choice_c3)

        # Choose random C4 branch
        branch_choice_c4 = self._get_random_branch_choice(len(self.C4.branches))
        print("branch choice c4:", branch_choice_c4)
        features_c4 = self.C4(features_c3, branch=branch_choice_c4)

        # Record route
        self.route[0] = branch_choice_c2
        self.route[1] = branch_choice_c3
        self.route[2] = branch_choice_c4

        for features in [features_c1, features_c2, features_c3, features_c4]:
            print("\nfeatures in stage")
            for idx, f in enumerate(features):
                print("feature[{}]:".format(idx), f.size())
       
        # Put final feature map in a list
        features = [features_c4] 
        # Forward thru RPN
        proposals, proposal_losses = self.rpn(images, features, targets)
        #for idx, p in enumerate(proposals):
        #    print("proposal[{}]:".format(idx), p)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        #print("\n\nbranch", branch_idx)
        #print("proposal_losses:")
        #for k, v in proposal_losses.items():
        #    print(k, v)
        #print("detector_losses:")
        #for k, v in detector_losses.items():
        #    print(k, v)


        #print("roi_head.x:", x.size())
        #print("roi_head.result:", len(result))


        # Update losses dictionary
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)


        #print("\n\nlosses:")
        #for k, v in losses.items():
        #    print(k, v)

        return losses

