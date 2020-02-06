#!/bin/bash

#python ./tools/train_net.py --config-file "configs/xview/faster_R101_C4__1x.yaml"
#python ./tools/train_net.py --config-file "configs/xview/faster_R50_C4__1x.yaml"
#python ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__1x.yaml"


python ./tools/train_net.py --config-file "configs/coco/faster_R50_C4__1x.yaml"
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_3-4-2.yaml"
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_resreg_up4x.yaml"
