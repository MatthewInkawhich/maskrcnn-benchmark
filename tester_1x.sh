#!/bin/bash

#python ./tools/test_net.py --config-file "configs/matt/faster_R101_C4__1x.yaml" --ckpt "logs/faster_R101_vanilla/model_final.pth"
python ./tools/test_net.py --config-file "configs/matt/faster_R101_C4__1x.yaml" --ckpt "logs/faster_R101_C4_stride8_nf/model_final.pth"
