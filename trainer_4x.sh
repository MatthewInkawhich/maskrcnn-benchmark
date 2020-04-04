#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_vanillacoco__4x.yaml"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R50_C4_stride16__4x.yaml" #--empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_dilation2__4x.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_dilation3__4x.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_roipool6__4x.yaml"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_anchorsizeA__4x.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_anchorsizeC__4x.yaml"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/fpn_R50__4x.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/fpn_R101__4x.yaml"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_control__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_primer__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C3C4__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C2C3C4__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4_dummy__4x.yaml" --empty-cache --skip-test

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4_primed__4x.yaml" --primer "out/xview/ewadaptive/ewa_R50_primer/model_pretrain_final.pth" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C3C4_primed__4x.yaml" --primer "out/xview/ewadaptive/ewa_R50_primer/model_pretrain_final.pth" --empty-cache --skip-test




### COCO sweeps
# Resolution
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_vanilla.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_stride4_np.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_stride8_np.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_stride32_np.yaml" --more-sizes

# Dilation
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_dilation2_np.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_dilation3_np.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_dilation4_np.yaml" --more-sizes

# Depth
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_3-4-2_np.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_3-4-8_np.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_3-4-14_np.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_3-4-20_np.yaml" --more-sizes

# Anchor Stride
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_resreg_down2x.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_resreg_keep1x.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_resreg_up2x.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_resreg_up4x.yaml"

# No Pretrain
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_vanilla_nopretrain.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_vanilla_nopretrain.yaml" --ckpt "out/coco/faster_R50_C4_vanilla_nopretrain/model_final.pth" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_fpn_nopretrain.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_fpn_nopretrain.yaml" --ckpt "out/coco/faster_R50_fpn_nopretrain/model_final.pth" --more-sizes


### COCO FPN
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_fpn.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_fpn_play.yaml"

#------------------------------------------------
### DDPP
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddpp_vanilla.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddpp_hourglass_skip.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddpp_cascade.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppss_cascade.yaml" --empty-cache

### DDPPv2
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_vanilla.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_cascadehead.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_cascadebodyhead.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2ss_cascadebodyhead.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_cascade_is.yaml" --empty-cache


#------------------------------------------------
### Custom
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/A_C4.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/B_C4.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/C_C4.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/vanilla_C4.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/E_C4.yaml" --empty-cache --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/F_C4.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/G_C4.yaml" --empty-cache


#------------------------------------------------
### Loss Weighting
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_A.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_B.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_C.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_D.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_E.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_F.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_G.yaml" --more-sizes


#------------------------------------------------
### Loss Weighting -- RPN Only
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_C4.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp1.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp2.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp3.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp4.yaml" --more-sizes
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp5.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp6.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp7.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp8.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp9.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp10.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_step1.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_step2.yaml"


#------------------------------------------------
### Strider
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/strider/faster_R50_strider_control_fpn.yaml" --more-sizes
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/strider/faster_R50_strider_vanilla.yaml" --empty-cache --more-sizes
