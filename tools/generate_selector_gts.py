# MatthewInkawhich

"""
This script generates binary GT maps for a pretrained EWA model for each
adaptive stage. 
"""

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

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
### Helpers
#####################################################################
def plot_image_and_gt(img, gt, num_branches, blend=True):
    # Define colors to use
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    # Normalize, permute channel dims and convert BGR --> RGB
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    img = img.permute(1, 2, 0)[:,:,[2,1,0]].to(torch.uint8).cpu().numpy()

    # Create gt_img
    gt_img_R = torch.zeros((gt.size(0), gt.size(1)))
    gt_img_G = torch.zeros((gt.size(0), gt.size(1)))
    gt_img_B = torch.zeros((gt.size(0), gt.size(1)))
    for branch in range(num_branches):
        mask = torch.eq(gt, branch)
        gt_img_R[mask] = colors[branch][0]
        gt_img_G[mask] = colors[branch][1]
        gt_img_B[mask] = colors[branch][2]

    gt_img = torch.stack([gt_img_R, gt_img_G, gt_img_B]).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    # Plot
    if blend:
        # Resize gt_img to same size as img
        gt_img = cv2.resize(gt_img, dsize=(img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
        # Create PIL Images
        gt_img_PIL = Image.fromarray(gt_img)
        img_PIL = Image.fromarray(img)
        # Blend
        blended_img = Image.blend(img_PIL, gt_img_PIL, alpha=0.75)
        plt.imshow(blended_img)
        plt.show()

    else:
        plt.figure(figsize=(10.8, 4.8))
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(img)

        colors_normalized = (np.array(colors) / 255).tolist()
        legend_elements = [Patch(facecolor=tuple(colors_normalized[0]), edgecolor=tuple(colors_normalized[0]), label="D=1"),
                           Patch(facecolor=tuple(colors_normalized[1]), edgecolor=tuple(colors_normalized[1]), label="D=2"),
                           Patch(facecolor=tuple(colors_normalized[2]), edgecolor=tuple(colors_normalized[2]), label="D=3")]
        plt.subplot(122)
        plt.title("Dilation Map")
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.imshow(gt_img)

    plt.show()
        




#####################################################################
### Generate Selector GTs
#####################################################################
def create_gt_map(intermediate_features):
    """
    This function takes a list intermediate_features, and uses magnitude
    of gradients to create a GT map for what branch is the best at each
    spatial location. Returns mask for each input in batch.
    """
    # Get gradients
    g = [a.grad for a in intermediate_features]
    # Stack tensor list into one tensor with size=[N x B X C x H x W]
    g = torch.stack(g).permute(1, 0, 2, 3, 4)
    #print("intermediate_grads:", g.shape)
    # Get sum of gradient magnitudes
    g = torch.abs(g)
    g = torch.sum(g, dim=2)
    # Here, g.shape=[N x B x H x W]
    #print("g:", g, g.shape)
    # Get element-wise min over branch (B) axis
    min_values, _ = torch.min(g, dim=1)
    #print("min_values:", min_values, min_values.shape)
    # Find all locations where B=1 element == min value at this position
    tiebreak_mask = torch.eq(g[:, 1, :, :], min_values)
    #print("tiebreak_mask:", tiebreak_mask, tiebreak_mask.shape)
    # Subtract 1 from B=1 element at the tiebreak locations
    g[:, 1, :, :][tiebreak_mask] -= 1.0
    #print("new g:", g, g.shape)
    # Take element-wise min over branch (B) axis again
    _, min_indices = torch.min(g, dim=1)
    #print("min_indices:", min_indices, min_indices.shape)
    return min_indices.to(torch.uint8)
    

def generate_selector_gts(model, data_loader, device):
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

        #print("loss_dict:")
        #for k, v in loss_dict.items():
        #    print(k, v)
        #print("losses:", losses)

        # Backprop gradients; intermediate_features tensors .grad are now populated
        losses.backward()

        # Craft GT selector map using intermediate_features gradients
        gt_map = create_gt_map(intermediate_features)

        #print("gt_map:", gt_map, gt_map.shape, gt_map.dtype)
        num_branches = len(intermediate_features)
        print("num_branches:", num_branches)
        # Iterate over batch
        for batch_idx in range(gt_map.size(0)):
            img = images.tensors[batch_idx]
            gt = gt_map[batch_idx]
            print("img:", img.shape)
            print("gt:", gt.shape)
            
            # Count number of each choice
            print("Counts:")
            for j in range(num_branches):
                c = gt[gt == j].shape[0]
                print("c"+str(j), c)

            # Plot image with GT mask 
            plot_image_and_gt(img, gt, num_branches, blend=False)


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
