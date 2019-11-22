# MatthewInkawhich

"""
This script generates binary GT maps for a pretrained EWA model for each
adaptive stage. 
"""

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


#####################################################################
### Generate Selector GTs
#####################################################################
def create_gt_map(intermediate_features):
    


def generate_selector_gts(
    model,
    data_loader,
    device,
):
    model.train()
    print("Dataset length:", len(data_loader.dataset))
    for iteration, (images, targets, idxs) in enumerate(data_loader, 0):
        # Images and targets to device
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        # Forward batch thru model will 'single_stage_all' option
        loss_dict, intermediate_features = model(images, targets, option="single_stage_all", adaptive_num=4)

        # Sum losses
        losses = sum(loss for loss in loss_dict.values())

        print("loss_dict:")
        for k, v in loss_dict.items():
            print(k, v)
        print("losses:", losses)

        # Backprop gradients; intermediate_features tensors .grad are now populated
        losses.backward()

        print(intermediate_features[0].shape, intermediate_features[1].shape, intermediate_features[2].shape)
        print("gradients:")
        print(intermediate_features[0].grad)

        g = intermediate_features[0].grad[0]
        #g = intermediate_features[0].grad
        g = torch.abs(g)
        g = torch.sum(g, dim=0)
        print(g, g.min(), g.max(), g.shape)

        # Craft GT selector map using intermediate_features gradients
        gt_map = create_gt_map(intermediate_features)

        exit()




#####################################################################
### Main
#####################################################################
def main():
    parser = argparse.ArgumentParser(description="Generate selector GTs for pretrained EWA model")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint to use, default is the latest checkpoint.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        exit("Error: This script only supports single GPU (due to gradient hooking).")
    #    torch.cuda.set_device(args.local_rank)
    #    torch.distributed.init_process_group(
    #        backend="nccl", init_method="env://"
    #    )
    #    synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # To avoid CUDA out of memory, lower train top_n
    cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = cfg.MODEL.EWADAPTIVE.GENERATE_SELECTOR_GTS_PRE_NMS_TOP_N
    cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN = cfg.MODEL.EWADAPTIVE.GENERATE_SELECTOR_GTS_POST_NMS_TOP_N
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    print(model)

    # Initialize optimizer
    optimizer = make_optimizer(cfg, model)

    # Load checkpoint
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    
    # temp
    model.check_sync()

    # Create data loader
    # Use train set, no horizontal/vertical flipping, must provide filename
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed, generate_selector_gts=True)

    # For all adaptive stages, forward all training images with 'all', hook
    # gradients, generate binary GT maps, and save to disk for use later
    generate_selector_gts(model, data_loader, cfg.MODEL.DEVICE)


if __name__ == "__main__":
    main()
