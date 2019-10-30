# MatthewInkawhich

# This class contains and defines the image-level adaptive r-cnn model

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone, build_resnet_stem, build_resnet_stage, build_ila_switch
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class ILAdaptiveRCNN(nn.Module):

    def __init__(self, cfg):
        super(ILAdaptiveRCNN, self).__init__()

        #self.cfg = cfg.clone()
        
        # Build ResNet stages
        self.C1 = build_resnet_stem(cfg)
        self.C2, C2_out_channels = build_resnet_stage(cfg, stage=2, in_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS)
        self.C3, C3_out_channels = build_resnet_stage(cfg, stage=3, in_channels=C2_out_channels)
        self.C4, C4_out_channels = build_resnet_stage(cfg, stage=4, in_channels=C3_out_channels)

        # Build switch
        self.num_branches = len(cfg.MODEL.ILADAPTIVE.C4)
        self.switch = build_ila_switch(C3_out_channels, self.num_branches)

        # Build RPN
        self.rpn = build_rpn(cfg, C4_out_channels)
        # Build ROI heads
        self.roi_heads = build_roi_heads(cfg, C4_out_channels)


    def forward(self, images, targets=None, option=0):
        # The 'option' arg defines what sub _forward we want to use (i.e. branches, switch)
        if option < 0 or option > 2:
            raise ValueError("option arg is out of valid range")
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Choose forwarding option
        if option == 0:
            return self._forward_inference(images)
        if option == 1:
            return self._forward_all_branches(images, targets)
        if option == 2:
            return self._forward_differential(images, targets, switch=False)
        if option == 3:
            return self._forward_switch(images, targets)
        if option == 4:
            return self._forward_differential(images, targets, switch=True)


    def _forward_inference(self, images):
        """
        This function is used for inference, when we want our trained
        switches to guide the adaptive processing.

        ** Only supports batch size of 1!
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)

        ### Process image data with stages
        features_c1 = self.C1(images.tensors)
        features_c2 = self.C2(features_c1)
        features_c3 = self.C3(features_c2)

        # Let switch module make prediction for what branch to take
        switch_preds = self.switch(features_c3).max(1)[1]
        print("switch_preds:", switch_preds)

        ### Images in the batch will have different recommended C4s
        ### so need to break up the batch
        predicted_boxlists = []
        # Loop over samples in batch
        for i in range(len(switch_preds)):
            # Compute final feature map from C4, make it a list
            features = [self.C4(features_c3[i].unsqueeze_(), branch=switch_preds[i])] 
            # Forward thru RPN
            proposals, proposal_losses = self.rpn(images[i], features, targets=None)
            # Forward thru RoI heads
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets=None)
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}

            # Append predictions for this image to our predicted_boxlists list
            predicted_boxlists.append(result)


        print("predicted_boxlists:", predicted_boxlists)
        exit()
        return predicted_boxlists


    def _forward_all_branches(self, images, targets):
        """
        This function is used for pretraining of branches and differential training
        stages. It returns a losses dictionary which contains separate entries for
        each loss component from each branch.
        """
        # Convert images input to image_list (if it isnt already)
        images = to_image_list(images)
        #print("images:", images.tensors.size())


        ### Process image data with stages
        features_c1 = self.C1(images.tensors)
        features_c2 = self.C2(features_c1)
        features_c3 = self.C3(features_c2)

        for features in [features_c1, features_c2, features_c3]:
            print("\nfeatures in stage")
            for idx, f in enumerate(features):
                print("feature[{}]:".format(idx), f.size())
       
        losses = {}

        # For each branch in C4 stage, make predicions
        for branch_idx in range(self.num_branches):
            # Compute final feature map from C4, make it a list
            features = [self.C4(features_c3, branch=branch_idx)] 
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


            # Update losses dictionary with losses from this branch
            for k, v in proposal_losses.items():
                losses["branch"+str(branch_idx)+"_"+k] = v
            for k, v in detector_losses.items():
                losses["branch"+str(branch_idx)+"_"+k] = v
                


        #print("\n\nlosses:")
        #for k, v in losses.items():
        #    print(k, v)

        return losses

