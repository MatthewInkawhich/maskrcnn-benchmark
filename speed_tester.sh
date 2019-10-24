#!/bin/bash

#python ./tools/test_net.py --config-file "configs/matt/faster_R101_C4__1x.yaml" --ckpt "logs/faster_R101_vanilla/model_final.pth"

python ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride4__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride4/model_final.pth" --speed-only
echo "STRIDE4"
python ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride8__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride8/model_final.pth" --speed-only
echo "STRIDE8"
python ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride16__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride16/model_final.pth" --speed-only
echo "STRIDE16"
python ./tools/test_net.py --config-file "configs/xview/faster_R101_C4_stride24__4x.yaml" --ckpt "out/xview/faster_R101_C4_stride24/model_0140000.pth" --speed-only
echo "STRIDE24"
