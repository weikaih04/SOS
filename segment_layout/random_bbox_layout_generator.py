import torch
from PIL import Image
import numpy as np
import pandas as pd
import random
import ast

def postprocess_bboxes(bbox_predictions, width, height):
    # Scale the normalized bbox predictions to image dimensions.
    processed_bboxes = []
    for bbox in bbox_predictions:
        xmin = bbox[0] * width
        ymin = bbox[1] * height
        xmax = bbox[2] * width
        ymax = bbox[3] * height
        processed_bboxes.append([xmin, ymin, xmax, ymax])
    return processed_bboxes

def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) for two boxes in normalized coordinates.
    Each bbox is in the format [x_min, y_min, x_max, y_max].
    """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

class RandomBoundingBoxLayoutGenerator():
    def __init__(self, device=None):
        # Determine device.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def generate(self, num_bboxes, width, height, min_box_scale=0.05, max_box_scale=0.6, max_overlap=0.5):
        """
        Generate num_bboxes random bounding boxes in a controlled way with limited overlap.
        The boxes are defined in normalized coordinates and then scaled to the given width and height.
        min_box_scale and max_box_scale define the fraction of the image that a box can take in width and height.
        max_overlap defines the maximum allowed Intersection over Union (IoU) between any two boxes.
        """
        bboxes = []
        max_attempts_total = num_bboxes * 50  # Maximum number of candidate generations.
        attempts = 0
        
        while len(bboxes) < num_bboxes and attempts < max_attempts_total:
            # Randomly select box width and height as a fraction of the full image.
            box_w = random.uniform(min_box_scale, max_box_scale)
            box_h = random.uniform(min_box_scale, max_box_scale)
            # Choose top-left coordinates so that the box fits within [0,1].
            x_min = random.uniform(0, 1 - box_w)
            y_min = random.uniform(0, 1 - box_h)
            x_max = x_min + box_w
            y_max = y_min + box_h + 0.3
            candidate_box = [x_min, y_min, x_max, y_max]
            
            # Check overlap with all previously accepted boxes.
            valid = True
            for existing_box in bboxes:
                if compute_iou(candidate_box, existing_box) > max_overlap:
                    valid = False
                    break
                    
            if valid:
                bboxes.append(candidate_box)
            attempts += 1

        if len(bboxes) < num_bboxes:
            print(f"Warning: Only generated {len(bboxes)} boxes with the given overlap constraints.")
            
        processed_bboxes = postprocess_bboxes(bboxes, width, height)
        return processed_bboxes