# MatthewInkawhich

# This class contains and defines the element-wise adaptive r-cnn model

import torch
from torch import nn
import random
import itertools

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
        #self.selector_list = []
        self.num_stages = 4
        self.adaptive_stages = []
        self.branch_counts = []
        # For each stage, if the stage has more than 1 branch, create a selector named 'branch{}'.format(stage_number)
        for stage_idx in range(len(ewa_stage_specs)):
            num_branches = len(ewa_stage_specs[stage_idx])
            self.branch_counts.append(num_branches)
            if num_branches > 1:
                selector = build_ewa_selector(out_channels[stage_idx], num_branches)
                name = "selector" + str(stage_idx + 2)
                self.add_module(name, selector)
                #self.selector_list.append(name)
                self.adaptive_stages.append("C" + str(stage_idx + 2))

        # Build RPN
        self.rpn = build_rpn(cfg, C4_out_channels)
        # Build ROI heads
        self.roi_heads = build_roi_heads(cfg, C4_out_channels)
        # Initialize route (elements correspond to the route taken thru each adaptive stage)
        self.route = [0, 0, 0]
        # Create list of routes (lists of ints corresponding to branch) that are possible
        expanded_branch_counts = [list(range(b)) for b in self.branch_counts]
        self.all_routes = list(itertools.product(*expanded_branch_counts))
        print("self.all_routes:", self.all_routes)


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



    def forward(self, images, targets=None, option="inference", iteration=0, adaptive_num=4):
        # The 'option' arg defines what sub _forward we want to use
        if self.training and targets is None:
            raise ValueError("Error: In training mode, targets should be passed")
        
        # Choose forwarding option
        if option == "inference":
            return self._forward_inference(images)
        if option == "pretrain":
            return self._forward_pretrain(images, targets, iteration)
        if option == "single_stage_all":
            return self._forward_single_stage_all(images, targets, adaptive_num)
        else:
            raise ValueError("Error: option arg is NOT valid!")



    def _forward_inference(self, images):
        """
        This function is used for inference, when we want our trained
        selectors to guide the adaptive processing.
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)
        #print("input:", images.tensors, images.tensors.shape)

        ### Process image data with stages
        features_C1 = self.C1(images.tensors)
        features_C2 = self.C2(features_C1, branch=0)
        features_C3 = self.C3(features_C2, branch=1)
        features_C4 = self.C4(features_C3, branch=1)

        features = [features_C4]
        #print("features_C1:", features_C1, features_C1.shape)
        #print("features_C1:", features_C2, features_C2.shape)
        #print("features_C1:", features_C3, features_C3.shape)
        #print("features_C4:", features_C4, features_C4.shape)

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
        features_C1 = self.C1(images.tensors)
        
        # Choose random C2 branch
        branch_choice_C2 = self._get_random_branch_choice(len(self.C2.branches), seed=iteration)
        #print("branch choice c2:", branch_choice_C2)
        features_C2 = self.C2(features_C1, branch=branch_choice_C2)

        # Choose random C3 branch
        branch_choice_C3 = self._get_random_branch_choice(len(self.C3.branches), seed=iteration+1)
        #print("branch choice c3:", branch_choice_C3)
        features_C3 = self.C3(features_C2, branch=branch_choice_C3)

        # Choose random C4 branch
        branch_choice_C4 = self._get_random_branch_choice(len(self.C4.branches), seed=iteration+2)
        #print("branch choice c4:", branch_choice_C4)
        features_C4 = self.C4(features_C3, branch=branch_choice_C4)

        #print("features_C4:", features_C4.shape)

        # Record route
        self.route[0] = branch_choice_C2
        self.route[1] = branch_choice_C3
        self.route[2] = branch_choice_C4
       
        # Put final feature map in a list
        features = [features_C4] 
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




    def _forward_single_stage_all(self, images, targets, adaptive_num):
        """
        This function is used for generating selector GT maps. For the selector at
        selector_idx, run all branches till prediction and return a loss dict with
        entries for losses from each branch. For all other adaptive stages, use
        branch=1 (dilation=2).
        """

        # Initialize losses dict
        losses = {}

        # Create stage lists
        selected_adaptive_stage = "C" + str(adaptive_num)
        before_stages = ["C" + str(i) for i in range(2, adaptive_num)]
        after_stages = ["C" + str(i) for i in range(adaptive_num + 1, self.num_stages + 1)]

        print("selected_adaptive_stage:", selected_adaptive_stage)
        print("before_stages:", before_stages)
        print("after_stages:", after_stages)
       

        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)
        #print(images.tensors.shape)
        #exit()

        # Forward pass thru stem (as this is never adaptive)
        features = self.C1(images.tensors)

        # Forward thru stages before adaptive stage C<adaptive_num>
        for stage_name in before_stages:
            print("forwarding thru ", stage_name)
            if stage_name in self.adaptive_stages:
                features = getattr(self, stage_name)(features, branch=1)
            else:
                features = getattr(self, stage_name)(features, branch=0)
        before_features = features

        # At this point, we're at the adaptive stage we care about
        # Forward all branches of this stage to loss
        intermediate_features = []
        num_branches = len(getattr(self, selected_adaptive_stage).branches)
        for curr_branch_idx in range(num_branches):
            print("Starting second phase, branch:", curr_branch_idx)
            # Forward thru selected adaptive branch
            features = getattr(self, selected_adaptive_stage)(before_features, branch=curr_branch_idx)
            features.retain_grad()
            # Store intermediate features to list to return later
            intermediate_features.append(features)
            # Iterate over after_stages
            for stage_name in after_stages:
                print("forwarding thru ", stage_name)
                if stage_name in self.adaptive_stages:
                    features = getattr(self, stage_name)(features, branch=1)
                else:
                    features = getattr(self, stage_name)(features, branch=0)

            # Enclose features in list
            features = [features]

            # Forward thru RPN
            proposals, proposal_losses = self.rpn(images, features, targets)

            # Forward thru RoI heads
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}

            # Add losses from this branch
            for k, v in proposal_losses.items():
                losses[k+str(curr_branch_idx)] = v
            for k, v in detector_losses.items():
                losses[k+str(curr_branch_idx)] = v

        
        return losses, intermediate_features
