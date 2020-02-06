#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_vanilla.yaml" --ckpt "out/coco/faster_R50_vanilla/model_final.pth" --more-sizes
#echo "VANILLA"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_stride4.yaml" --ckpt "out/coco/faster_R50_stride4/model_final.pth" --more-sizes
#echo "STRIDE4"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_stride8.yaml" --ckpt "out/coco/faster_R50_stride8/model_final.pth" --more-sizes
#echo "STRIDE8"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_stride32.yaml" --ckpt "out/coco/faster_R50_stride32/model_final.pth" --more-sizes
#echo "STRIDE32"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_dilation2.yaml" --ckpt "out/coco/faster_R50_dilation2/model_final.pth" --more-sizes
#echo "DILATION2"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_dilation3.yaml" --ckpt "out/coco/faster_R50_dilation3/model_final.pth" --more-sizes
#echo "DILATION3"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_dilation4.yaml" --ckpt "out/coco/faster_R50_dilation4/model_final.pth" --more-sizes
#echo "DILATION4"


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-2.yaml" --ckpt "out/coco/faster_R50_3-4-2/model_final.pth" --more-sizes
#echo "3-4-2"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-8.yaml" --ckpt "out/coco/faster_R50_3-4-8/model_final.pth" --more-sizes
#echo "3-4-8"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-14.yaml" --ckpt "out/coco/faster_R50_3-4-14/model_final.pth" --more-sizes
#echo "3-4-14"
#python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_C4_3-4-20.yaml" --ckpt "out/coco/faster_R50_3-4-20/model_final.pth" --more-sizes
#echo "3-4-20"


python -m torch.distributed.launch --nproc_per_node=4 ./tools/test_net.py --config-file "configs/coco/faster_R50_fpn.yaml" --ckpt "out/coco/faster_R50_fpn/model_final.pth" --more-sizes
echo "FPN"
