#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_middleks_3-4-8__4x.yaml" #--empty-cache
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_A__4x.yaml" --empty-cache
