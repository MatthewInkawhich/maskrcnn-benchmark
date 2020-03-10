# MatthewInkawhich

"""
Procedure to probe aspects of FG anchor target matches and anchor-wise loss contribution.
Similar to train_net.py, but we are not updating model parameters.

Only supports single GPU.
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train, do_pretrain_ewadaptive
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config


########################################################################################
### Probing procedure for Generalized R-CNN
########################################################################################
def do_probe(
    model,
    data_loader,
    device,
    savepath,
):
    max_iter = len(data_loader)
    print("max_iter:", max_iter)
    # Model must be in train mode to compute losses
    model.train()

    # Set first flag
    first_iter = True

    # Iterate over iteration-based data loader
    for iteration, (images, targets, _) in enumerate(data_loader):
        print("iteration:", iteration)

        # Save GT bboxes
        #curr_gt_boxes = torch.cat([bl.bbox for bl in targets], dim=0)

        #if first_iter:
        #    gt_boxes = curr_gt_boxes
        #    first_iter = False
        #else:
        #    gt_boxes = torch.cat((gt_boxes, curr_gt_boxes), dim=0)
        #continue

        # Load images and targets to device
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # Forward data thru model with no gradient computation
        with torch.no_grad():
            curr_fg_anchor_stats = model(images, targets, probe=True)

        print("curr_fg_anchor_stats:", curr_fg_anchor_stats.shape)

        # Add to total
        if first_iter:
            total = curr_fg_anchor_stats
            first_iter = False
        else:
            total = torch.cat((total, curr_fg_anchor_stats), dim=0)

    # When done with iterations, save to disk
    torch.save(total, savepath)

    #torch.save(gt_boxes, "./junk/gt_boxes_train50000.pt")
        




########################################################################################
### Main
########################################################################################
def main():
    # Handle CL argument parsing
    parser = argparse.ArgumentParser(description="Probe Loss")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--weights",
        default="",
        type=str,
        help="path to trained checkpoint"
    )
    parser.add_argument(
        "--savepath",
        default="./junk/probe_loss_out.pt",
        type=str,
        help="path to out file"
    )
    args = parser.parse_args()



    ## TMP
    #loaded_total = torch.load(args.savepath)
    #print(loaded_total, loaded_total.shape)
    #exit()



    # Assemble config from file and CL options
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Initialize logger
    logger = setup_logger("maskrcnn_benchmark", "", get_rank())

    # Construct model
    model = build_detection_model(cfg)
    print(model)
    print("params:", sum(p.numel() for p in model.parameters()))
    #exit()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # Load pretrained base network weights
    checkpointer = DetectronCheckpointer(cfg, model)
    extra_checkpoint_data = checkpointer.load(args.weights)

    # Make data loader
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    # Run probe
    do_probe(
        model,
        data_loader,
        device,
        args.savepath
    )


if __name__ == "__main__":
    main()
