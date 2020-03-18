#!/bin/bash

#python ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C4/model_pretrain_final.pth"

#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn/model_final.pth" --more-sizes



#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_C4.yaml" --ckpt "out/coco/rpn_only/rpn_R50_C4/model_final.pth" --more-sizes
#echo "C4"

#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn/model_final.pth" --more-sizes
#echo "FPN"

#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp1.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp1/model_final.pth" --more-sizes
#echo "FPN_EXP1"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp2.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp2/model_final.pth" --more-sizes
#echo "FPN_EXP2"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp3.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp3/model_final.pth" --more-sizes
#echo "FPN_EXP3"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp4.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp4/model_final.pth" --more-sizes
#echo "FPN_EXP4"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp5.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp5/model_final.pth" --more-sizes
#echo "FPN_EXP5"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp6.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp6/model_final.pth" --more-sizes
#echo "FPN_EXP6"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp7.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp7/model_final.pth" --more-sizes
#echo "FPN_EXP7"
python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp8.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp8/model_final.pth" --more-sizes
echo "FPN_EXP8"
python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp9.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp9/model_final.pth" --more-sizes
echo "FPN_EXP9"
python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_exp10.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_exp10/model_final.pth" --more-sizes
echo "FPN_EXP10"
#python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_step1.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_step1/model_final.pth" --more-sizes
#echo "FPN_STEP1"
python ./tools/test_net.py --config-file "configs/coco/rpn_only/rpn_R50_fpn_step2.yaml" --ckpt "out/coco/rpn_only/rpn_R50_fpn_step2/model_final.pth" --more-sizes
echo "FPN_STEP2"
