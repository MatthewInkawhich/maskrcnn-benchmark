#!/bin/bash

#python ./tools/train_net.py --config-file "configs/xview/faster_R101_C4__1x.yaml"
#python ./tools/train_net.py --config-file "configs/xview/faster_R50_C4__1x.yaml"
#python ./tools/train_net.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__1x.yaml"


#python ./tools/train_net.py --config-file "configs/coco/faster_R50_C4__1x.yaml"
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_3-4-2.yaml"
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_C4_resreg_up4x.yaml"


#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddpp_vanilla.yaml" --empty-cache
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddpp_hourglass_skip.yaml" --empty-cache
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddpp_cascade.yaml" --empty-cache
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppss_cascade.yaml" --empty-cache

#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_vanilla.yaml" --empty-cache
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_cascadehead.yaml" --empty-cache
#python ./tools/train_net.py --config-file "configs/coco/faster_R50_ddppv2_cascadebodyhead.yaml" --empty-cache


# Custom
python ./tools/train_net.py --config-file "configs/coco/custom/C4_vanilla.yaml" --empty-cache
