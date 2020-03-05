#!/bin/bash

python ./tools/probe_loss.py --config-file "configs/coco/probe/faster_R50_C4_vanilla_probe.yaml" --weights "out/coco/faster_R50_vanilla/model_final.pth"
