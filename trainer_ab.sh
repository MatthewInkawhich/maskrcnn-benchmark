#!/bin/bash

### Custom
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/B_C4.yaml" --empty-cache --more-sizes
echo "Finished B"
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_net.py --config-file "configs/coco/custom/A_C4.yaml" --empty-cache --more-sizes
echo "Finished A"
