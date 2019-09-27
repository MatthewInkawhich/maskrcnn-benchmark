#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/matt/faster_rcnn_R_101_C4_4x_stride8.yaml" --ckpt "logs/faster_R101_stride8/model_final.pth"
