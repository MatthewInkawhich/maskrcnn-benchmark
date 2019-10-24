# MatthewInkawhich

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json

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
# Draw boxes on the images
# Note: img is a PIL ImageDraw.draw
def draw_boxes(draw, boxes, labels=None, scores=None, box_color=(255,130,0), text_color=(180,255,0), linesize=3):
    for i in range(len(boxes)):
        # Draw box
        xmin, ymin, xmax, ymax = boxes[i]
        for j in range(linesize):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline=box_color)
        # Draw text for this box if labels were provided
        if labels:
            draw.text((xmin+linesize, ymin+linesize), str(labels[i]), fill=text_color)


# Read labels.json file, and return mappings: (1) contiguous_lbl --> int_lbl, (2) int_lbl --> str_lbl
def get_label_mappings(label_path):
    # Read labels.json
    with open(label_path) as json_file:
        json_mapping = json.load(json_file)
    # Create new mapping
    contiguous_to_orig_map = {}
    orig_to_str_map = {}
    contiguous_lbl = 1
    for int_lbl, str_lbl in json_mapping.items():
        contiguous_to_orig_map[contiguous_lbl] = int(int_lbl)
        orig_to_str_map[int(int_lbl)] = str_lbl
        contiguous_lbl += 1
    return contiguous_to_orig_map, orig_to_str_map

# Generate str_labels list using the orig labels and a mapping dict
def generate_str_labels(orig_labels, mapping):
    str_labels = []
    for i in range(len(orig_labels)):
        str_labels.append(mapping[orig_labels[i]])
    return str_labels

# Get the GT box and label annotations corresponding to the image_path
def get_gt_labels(annotation_path, image_path):
    # Read annotations json
    with open(annotation_path) as json_file:
        head = json.load(json_file)
    # Get image_id that corresponds to the image_path
    image_file_name = image_path.split('/')[-1]
    image_id = -1
    for image_head in head['images']:
        if image_head['file_name'] == image_file_name:
            image_id = image_head['id']
            break
    # Check to see if we got a valid image_id
    if image_id == -1:
        print("Failed to find an image_id matching the file given!")
        return [], []
    # Now gather all box annotations in image image_id
    gt_boxes = []
    gt_labels = []
    for annotation_head in head['annotations']:
        if annotation_head['image_id'] == image_id:
            topleft_x, topleft_y, w, h = annotation_head['bbox']
            gt_boxes.append([topleft_x, topleft_y, topleft_x + w, topleft_y + h])
            gt_labels.append(annotation_head['category_id'])
    # Return boxes and labels
    return gt_boxes, gt_labels




##################################
### MAIN
##################################
def main():
    # Configure
    XVIEW = True
    PLOT_GT = False
    PLOT_PREDICTIONS = True
    TEXT = False

    # Set paths
    if XVIEW:
        CONFIG_FILE_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'configs', 'xview', 'faster_R101_C4_stride4__4x.yaml')
        OUT_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'out', 'xview', 'faster_R101_C4_stride4')
        #CONFIG_FILE_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'configs', 'xview', 'faster_R101_C4_stride24__4x.yaml')
        #OUT_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'out', 'xview', 'faster_R101_C4_stride24')
        CHECKPOINT_PATH = os.path.join(OUT_PATH, 'model_final.pth')
        LABEL_PATH = os.path.join(OUT_PATH, 'labels.json')
        DATA_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'datasets', 'xView-coco-600')
        ANNOTATION_PATH = os.path.join(DATA_PATH, 'annotations', 'val_full.json')
        #IMAGE_PATH = os.path.join(DATA_PATH, 'val_images', 'img_97_14_rot0.jpg')
        IMAGE_PATH = os.path.join(DATA_PATH, 'val_images', 'img_322_30_rot0.jpg')
        DRAW_THRESH = 0.6
    else: # COCO
        CONFIG_FILE_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'configs', 'coco', 'faster_R101_C4_vanilla__4x.yaml')
        OUT_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'out', 'coco', 'faster_R101_vanilla')
        CHECKPOINT_PATH = os.path.join(OUT_PATH, 'model_final.pth')
        LABEL_PATH = os.path.join(OUT_PATH, 'labels.json')
        DATA_PATH = os.path.join(os.path.expanduser('~'), 'WORK', 'maskrcnn-benchmark', 'datasets', 'coco')
        ANNOTATION_PATH = os.path.join(DATA_PATH, 'annotations', 'instances_val2017.json')
        IMAGE_PATH = os.path.join(DATA_PATH, 'val2017', '000000000139.jpg')
        DRAW_THRESH = 0.6



    # Read image to PIL Image
    pil_img = Image.open(IMAGE_PATH)
    pil_img_draw = pil_img.convert("RGB")
    draw = ImageDraw.Draw(pil_img_draw)

    # Load label_mapping from labels.json    
    contiguous_to_orig_map, orig_to_str_map = get_label_mappings(LABEL_PATH)

    # Read and draw GT boxes
    if PLOT_GT:
        gt_boxes, gt_labels = get_gt_labels(ANNOTATION_PATH, IMAGE_PATH)
        gt_str_labels = generate_str_labels(gt_labels, orig_to_str_map) if TEXT else []
        draw_boxes(draw=draw, boxes=gt_boxes, labels=gt_str_labels, box_color=(0,0,255), text_color=(0,0,255))

    # Generate and draw predictions
    if PLOT_PREDICTIONS:
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

        # Convert to tensor and perform transforms
        img = F.to_tensor(pil_img)
        if cfg.INPUT.TO_BGR255:
            img = img[[2, 1, 0]] * 255
        img = F.normalize(img, mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        img = img.unsqueeze(0)
        print("img:", img.shape)

        # Convert to ImageList (BatchCollator typically does this for us)
        # so model can ingest it
        img_input = to_image_list(img)
        #img_input.to(cfg.MODEL.DEVICE)

        # Forward pass img_input thru model
        # This returns a BoxList object with the detections
        with torch.no_grad():
            output = model(img_input)

        # Separate predictions into separate variables
        out_boxlist = output[0]
        predicted_boxes = out_boxlist.bbox
        predicted_scores = out_boxlist.get_field('scores')
        predicted_labels = out_boxlist.get_field('labels')

        # Filter out predictions under the DRAW_THRESH
        draw_mask = predicted_scores >= DRAW_THRESH
        predicted_boxes = predicted_boxes[draw_mask].tolist()
        predicted_scores = predicted_scores[draw_mask].tolist()
        predicted_labels = predicted_labels[draw_mask].tolist()
        predicted_orig_labels = [contiguous_to_orig_map[int(x)] for x in predicted_labels]

        # Create str_label list to plot
        str_labels = generate_str_labels(predicted_orig_labels, orig_to_str_map) if TEXT else []

        # Draw prediction boxes
        draw_boxes(draw=draw, boxes=predicted_boxes, labels=str_labels, box_color=(255,130,0), text_color=(180,255,0))


    # Show image
    plt.imshow(pil_img_draw)
    plt.show()



if __name__ == "__main__":
    main()
