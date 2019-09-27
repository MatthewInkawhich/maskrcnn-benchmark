#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=3 ./tools/train_net.py --config-file "configs/matt/faster_rcnn_R_101_C4_3x.yaml"
