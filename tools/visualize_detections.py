# MatthewInkawhich

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list 


##################################
### HELPERS
##################################
# Draw boxes on the images, return modified img
# Note: img is a PIL image
def draw_detections(img, boxes, labels, scores=None, box_color=(255,130,0), text_color=(180,255,0), linesize=3):
    rgb_img = img.convert("RGB")
    draw = ImageDraw.Draw(rgb_img)
    #fnt = ImageFont.truetype("arial.ttf", 18)
    w, h = (img.width, img.height)
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
        for j in range(linesize):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline=box_color)
        draw.text((xmin+linesize, ymin+linesize), str(labels[i].item()), fill=text_color)
    return rgb_img



def main():
    # Set paths
    CONFIG_FILE_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'configs', 'coco', 'faster_R101_C4_stride16__4x.yaml')
    CHECKPOINT_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'out', 'coco', 'faster_R101_C4_stride16_nopretrain', 'model_final.pth')
    IMAGE_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'datasets', 'coco', 'val2017', '000000000139.jpg')
    PLOT_GT = False
    DRAW_THRESH = 0.5

    # Load configs
    cfg.merge_from_file(CONFIG_FILE_PATH)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Build model
    model = build_detection_model(cfg)
    #model.to(cfg.MODEL.DEVICE)
    model.eval()

    # Load model checkpoint
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(CHECKPOINT_PATH, use_latest=False)


    # Read image to PIL Image
    pil_img = Image.open(IMAGE_PATH)
    # Convert to tensor and perform transforms
    img = F.to_tensor(pil_img)
    if cfg.INPUT.TO_BGR255:
        img = img[[2, 1, 0]] * 255
    img = F.normalize(img, mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    img = img.unsqueeze(0)
    print("img:", img, img.shape)

    # Convert to ImageList (BatchCollator typically does this for us)
    # so model can ingest it
    img_input = to_image_list(img)
    #img_input.to(cfg.MODEL.DEVICE)
    print("img_input:", img_input, img_input.tensors)

    # Forward pass img_input thru model
    # This returns a BoxList object with the detections
    with torch.no_grad():
        output = model(img_input)

    out_boxlist = output[0]
    # Separate predictions into separate variables
    predicted_boxes = out_boxlist.bbox
    predicted_scores = out_boxlist.get_field('scores')
    predicted_labels = out_boxlist.get_field('labels')


    # Filter out predictions under the DRAW_THRESH
    draw_mask = predicted_scores >= DRAW_THRESH
    predicted_boxes = predicted_boxes[draw_mask]
    predicted_scores = predicted_scores[draw_mask]
    predicted_labels = predicted_labels[draw_mask]
    #print("predicted_boxes:", predicted_boxes)
    #print("predicted_scores:", predicted_scores)
    #print("predicted_labels:", predicted_labels)

    pil_img = draw_detections(img=pil_img, boxes=predicted_boxes, labels=predicted_labels)
    plt.imshow(pil_img)
    plt.show()

    exit()

    # Plot image
    im2show = img[0].permute(1,2,0)
    im2show = (im2show - im2show.min()) / (im2show.max() - im2show.min())
    plt.imshow(im2show)
    plt.show()



if __name__ == "__main__":
    main()
