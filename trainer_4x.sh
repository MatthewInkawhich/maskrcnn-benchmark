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
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_stride4.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_stride8.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_stride32.yaml"

# Dilation
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_dilation2_np.yaml"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_dilation3_np.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/nopretrain/faster_R50_C4_dilation4_np.yaml"

# Depth
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_3-4-2.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_3-4-8.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_3-4-14.yaml"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_3-4-20.yaml"

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


