#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python ./tools/generate_selector_gts.py --config-file "configs/xview/ewadaptive/ewa_R50_C4__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C4/model_pretrain_final.pth"
#CUDA_VISIBLE_DEVICES=3 python ./tools/generate_selector_gts.py --config-file "configs/xview/ewadaptive/ewa_R50_C3C4__4x.yaml" --ckpt "out/xview/ewadaptive/ewa_R50_C3C4/model_pretrain_final.pth"
