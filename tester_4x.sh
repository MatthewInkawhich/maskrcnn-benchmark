#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_vanilla.yaml" --ckpt "out/coco/faster_R50_vanilla/model_final.pth" --more-sizes
#echo "VANILLA"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_stride4.yaml" --ckpt "out/coco/faster_R50_stride4/model_final.pth" --more-sizes
#echo "STRIDE4"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_stride8.yaml" --ckpt "out/coco/faster_R50_stride8/model_final.pth" --more-sizes
#echo "STRIDE8"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_stride32.yaml" --ckpt "out/coco/faster_R50_stride32/model_final.pth" --more-sizes
#echo "STRIDE32"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_dilation2.yaml" --ckpt "out/coco/faster_R50_dilation2/model_final.pth" --more-sizes
#echo "DILATION2"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_dilation3.yaml" --ckpt "out/coco/faster_R50_dilation3/model_final.pth" --more-sizes
#echo "DILATION3"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_dilation4.yaml" --ckpt "out/coco/faster_R50_dilation4/model_final.pth" --more-sizes
#echo "DILATION4"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-2.yaml" --ckpt "out/coco/faster_R50_3-4-2/model_final.pth" --more-sizes
#echo "3-4-2"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-8.yaml" --ckpt "out/coco/faster_R50_3-4-8/model_final.pth" --more-sizes
#echo "3-4-8"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-14.yaml" --ckpt "out/coco/faster_R50_3-4-14/model_final.pth" --more-sizes
#echo "3-4-14"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-20.yaml" --ckpt "out/coco/faster_R50_3-4-20/model_final.pth" --more-sizes
#echo "3-4-20"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_fpn.yaml" --ckpt "out/coco/faster_R50_fpn/model_final.pth" --more-sizes
#echo "FPN"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_vanilla_nopretrain.yaml" --ckpt "out/coco/faster_R50_vanilla_nopretrain/model_final.pth" --more-sizes
#echo "C4 nopretrain"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_fpn_nopretrain.yaml" --ckpt "out/coco/faster_R50_fpn_nopretrain/model_final.pth" --more-sizes
#echo "FPN nopretrain"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddpp_vanilla.yaml" --ckpt "out/coco/faster_R50_ddpp_vanilla/model_final.pth" --more-sizes
#echo "DDPP vanilla"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddpp_hourglass_skip.yaml" --ckpt "out/coco/faster_R50_ddpp_hourglass_skip/model_final.pth" --more-sizes
#echo "DDPP hourglass-skip"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddpp_cascade.yaml" --ckpt "out/coco/faster_R50_ddpp_cascade/model_final.pth" --more-sizes
#echo "DDPP cascade"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddppss_cascade.yaml" --ckpt "out/coco/faster_R50_ddppss_cascade/model_final.pth" --more-sizes
#echo "DDPP-SS cascade"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddppv2_vanilla.yaml" --ckpt "out/coco/faster_R50_ddppv2_vanilla/model_final.pth" --more-sizes
#echo "DDPPv2 vanilla"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddppv2_cascadehead.yaml" --ckpt "out/coco/faster_R50_ddppv2_cascadehead/model_final.pth" --more-sizes
#echo "DDPPv2 cascadehead"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddppv2_cascadebodyhead.yaml" --ckpt "out/coco/faster_R50_ddppv2_cascadebodyhead/model_final.pth" --more-sizes
#echo "DDPPv2 cascadebodyhead"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddppv2ss_cascadebodyhead.yaml" --ckpt "out/coco/faster_R50_ddppv2ss_cascadebodyhead/model_final.pth" --more-sizes
#echo "DDPPv2ss cascadebodyhead"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_ddppv2_cascade_is.yaml" --ckpt "out/coco/faster_R50_ddppv2_cascade_is/model_final.pth" --more-sizes
#echo "DDPPv2 cascade IS"

#--------------------------------------- CUSTOM
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/custom/F_C4.yaml" --ckpt "out/coco/custom/F_C4/model_final.pth" --more-sizes
#echo "F_C4"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/custom/G_C4.yaml" --ckpt "out/coco/custom/G_C4/model_final.pth" --more-sizes
#echo "G_C4"



python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_vanilla.yaml" --ckpt "out/coco/faster_R50_vanilla/model_final.pth" --more-sizes
echo "C4"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_fpn.yaml" --ckpt "out/coco/faster_R50_fpn/model_final.pth" --more-sizes
echo "FPN"

python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_A.yaml" --ckpt "out/coco/loss_weighting/faster_R50_fpn_lw_exp_A/model_final.pth" --more-sizes
echo "FPN LW EXP A"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_B.yaml" --ckpt "out/coco/loss_weighting/faster_R50_fpn_lw_exp_B/model_final.pth" --more-sizes
echo "FPN LW EXP B"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/loss_weighting/faster_R50_fpn_lw_exp_C.yaml" --ckpt "out/coco/loss_weighting/faster_R50_fpn_lw_exp_C/model_final.pth" --more-sizes
echo "FPN LW EXP C"
