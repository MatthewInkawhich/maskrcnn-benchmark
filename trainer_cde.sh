#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/E_C4.yaml" --empty-cache --more-sizes
echo "Finished E"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/vanilla_C4.yaml" --empty-cache --more-sizes
echo "Finished vanilla"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/C_C4.yaml" --empty-cache --more-sizes
echo "Finished C"
