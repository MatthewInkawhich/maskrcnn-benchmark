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
        # For each stage, if the stage has more than 1 branch, create a selector named 'branch{}'.format(stage_number)
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
        self.route = [0, 0, 0]
        # Initialize synced flag, this represents whether or not the weights are synced
        self.synced = True


    def _get_random_branch_choice(self, num_branches, seed=0):
        """
        Helper function to generate random branch index choice for pretraining
        """
        random.seed(seed)

        branch_choice = 0
        if num_branches > 1:
            branch_choice = random.randint(0, num_branches - 1)
        return branch_choice



    def sync_weights(self):
        """
        Syncs weights across branches for all submodules according to self.route tuple.
        """
        # Sync C2
        if len(self.C2.branches) > 1:
            # Multiple branches, must sync according to route
            for branch_name in self.C2.branches:
                updated_branch_name = "branch" + str(self.route[0])
                # Replace all branches that are not the updated_branch weights with the updated_branch's weights
                if branch_name != updated_branch_name:
                    # Set this branch's weights equal to the updated_branch's weights
                    #print("Setting {} weights equal to {} weights".format(branch_name, updated_branch_name))
                    getattr(self.C2, branch_name).load_state_dict(getattr(self.C2, updated_branch_name).state_dict())

        # Sync C3
        if len(self.C3.branches) > 1:
            # Multiple branches, must sync according to route
            for branch_name in self.C3.branches:
                updated_branch_name = "branch" + str(self.route[1])
                # Replace all branches that are not the updated_branch weights with the updated_branch's weights
                if branch_name != updated_branch_name:
                    # Set this branch's weights equal to the updated_branch's weights
                    #print("Setting {} weights equal to {} weights".format(branch_name, updated_branch_name))
                    getattr(self.C3, branch_name).load_state_dict(getattr(self.C3, updated_branch_name).state_dict())

        # Sync C4
        if len(self.C4.branches) > 1:
            # Multiple branches, must sync according to route
            for branch_name in self.C4.branches:
                updated_branch_name = "branch" + str(self.route[2])
                # Replace all branches that are not the updated_branch weights with the updated_branch's weights
                if branch_name != updated_branch_name:
                    # Set this branch's weights equal to the updated_branch's weights
                    #print("Setting {} weights equal to {} weights".format(branch_name, updated_branch_name))
                    getattr(self.C4, branch_name).load_state_dict(getattr(self.C4, updated_branch_name).state_dict())



    def check_sync(self):
        #print("comparing branch0 and branch1")
        synced = True
        for pname0, p0 in self.C4.branch0.named_parameters():
            for pname1, p1 in self.C4.branch1.named_parameters():
                if pname0 == pname1:
                    #print("\n" + pname0)
                    if p0.data.ne(p1.data).sum() > 0:
                        synced = False
                        break
                        #print("NOT EQUAL!!")
                    #else:
                        #print("EQUAL")
               
        #print("\n\ncomparing branch1 and branch2")
        for pname1, p1 in self.C4.branch1.named_parameters():
            for pname2, p2 in self.C4.branch2.named_parameters():
                if pname1 == pname2:
                    #print("\n" + pname1)
                    if p1.data.ne(p2.data).sum() > 0:
                        synced = False
                        #print("NOT EQUAL!!")
                    #else:
                    #    print("EQUAL")

        if synced:
            print("Sync check: PASS")
        else:
            print("Sync check: FAIL")



    def forward(self, images, targets=None, option="inference", iteration=0, selector_idx=0):
        # The 'option' arg defines what sub _forward we want to use
        if option not in self.forward_options:
            raise ValueError("Error: option arg is NOT valid! Must choose one of self.forward_options")
        if self.training and targets is None:
            raise ValueError("Error: In training mode, targets should be passed")
        
        # Choose forwarding option
        if option == "inference":
            return self._forward_inference(images)
        if option == "pretrain":
            return self._forward_pretrain(images, targets, iteration)
        if option == "generate_selector_gt":
            return self._forward_generate_selector_gt(images, targets, selector_idx)



    def _forward_inference(self, images):
        """
        This function is used for inference, when we want our trained
        selectors to guide the adaptive processing.
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)

        ### Process image data with stages
        features_c1 = self.C1(images.tensors)
        features_c2 = self.C2(features_c1, branch=1)
        features_c3 = self.C3(features_c2, branch=1)
        features = [self.C4(features_c3, branch=1)] 

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



    def _forward_pretrain(self, images, targets, iteration):
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
        branch_choice_c2 = self._get_random_branch_choice(len(self.C2.branches), seed=iteration)
        #print("branch choice c2:", branch_choice_c2)
        features_c2 = self.C2(features_c1, branch=branch_choice_c2)

        # Choose random C3 branch
        branch_choice_c3 = self._get_random_branch_choice(len(self.C3.branches), seed=iteration+1)
        #print("branch choice c3:", branch_choice_c3)
        features_c3 = self.C3(features_c2, branch=branch_choice_c3)

        # Choose random C4 branch
        branch_choice_c4 = self._get_random_branch_choice(len(self.C4.branches), seed=iteration+2)
        #print("branch choice c4:", branch_choice_c4)
        features_c4 = self.C4(features_c3, branch=branch_choice_c4)

        # Record route
        self.route[0] = branch_choice_c2
        self.route[1] = branch_choice_c3
        self.route[2] = branch_choice_c4

        #for features in [features_c1, features_c2, features_c3, features_c4]:
        #    print("\nfeatures in stage")
        #    for idx, f in enumerate(features):
        #        print("feature[{}]:".format(idx), f.size())
       
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


        # Update losses dictionary
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)


        #print("\n\nlosses:")
        #for k, v in losses.items():
        #    print(k, v)

        return losses



    def _forward_generate_selector_gt(self, images, targets, selector_idx):
        """
        This function is used for generating selector GT maps. For the selector at
        selector_idx, run all branches till prediction and return a loss dict with
        entries for losses from each branch. For all other adaptive stages, use
        branch=1 (dilation=2).
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)

        ### Process image data with stages
        features_c1 = self.C1(images.tensors)
        features_c2 = self.C2(features_c1, branch=1)
        features_c3 = self.C3(features_c2, branch=1)
        features = [self.C4(features_c3, branch=1)] 

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
