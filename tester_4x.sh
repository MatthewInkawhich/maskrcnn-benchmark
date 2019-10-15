#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride16__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride16/model_final.pth"
