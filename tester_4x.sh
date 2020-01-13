#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride4__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride4/model_final.pth"
#echo "STRIDE4"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride8__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride8/model_final.pth"
#echo "STRIDE8"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride16__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride16/model_final.pth"
#echo "STRIDE16"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride24__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride24/model_0140000.pth"
#echo "STRIDE24"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_middleks_3-4-8__4x.yaml" --ckpt "out/xview/faster_R101_C4_middleks_3-4-8/model_0100000.pth"
#echo "MIDDLE_KS:3-4-8"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_middleks_3-4-16__4x.yaml" --ckpt "out/xview/faster_R101_C4_middleks_3-4-16/model_0100000.pth"
#echo "MIDDLE_KS:3-4-16"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_middleks_3-4-23__4x.yaml" --ckpt "out/xview/faster_R101_C4_middleks_3-4-23/model_0100000.pth"
#echo "MIDDLE_KS:3-4-23"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_roipool22__4x.yaml" --ckpt "out/xview/faster_R101_C4_roipool22/model_final.pth"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C4/model_pretrain_final.pth"
#echo "^C4"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C3C4__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C3C4/model_pretrain_final.pth"
#echo "^C3C4"

python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C2C3C4__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C2C3C4/model_pretrain_final.pth"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_control__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_control/model_pretrain_final.pth"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_primer__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_primer/model_pretrain_final.pth"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4_primed__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C4_primed/model_pretrain_final.pth"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C3C4_primed__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C3C4_primed/model_pretrain_final.pth"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R50_C4_dummy__4x.yaml"
