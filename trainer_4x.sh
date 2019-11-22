#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R50_C4_stride16__4x.yaml" #--empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_dilation2__4x.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/faster_R101_C4_dilation3__4x.yaml" --empty-cache
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/fpn_R50__4x.yaml"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/fpn_R101__4x.yaml"

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_control__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_primer__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C3C4__4x.yaml" --empty-cache --skip-test
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4_dummy__4x.yaml" --empty-cache --skip-test

