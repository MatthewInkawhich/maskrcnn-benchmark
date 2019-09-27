#!/bin/bash

python ./tools/test_net.py --config-file "configs/matt/faster_rcnn_R_101_C4_4x_vanilla.yaml" --ckpt "logs/faster_R101_vanilla/model_final.pth"
