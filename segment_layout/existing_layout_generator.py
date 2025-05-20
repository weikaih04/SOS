import torch
from PIL import Image
import numpy as np
import pandas as pd
import random
import ast

def calculate_image_areas(image_paths):
    components = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
            components.append((width, height))
    return components

def compute_similarity(component, box):
    """
    Compute similarity score based on area and aspect ratio.
    """
    # Unpack dimensions
    comp_h, comp_w = component
    box_h = box[2] - box[0]
    box_w = box[3] - box[1]

    # Calculate areas and aspect ratios
    comp_area = comp_h * comp_w
    box_area = box_h * box_w

    comp_aspect_ratio = comp_h / comp_w
    box_aspect_ratio = box_h / box_w

    # Area difference (normalized)
    area_diff = abs(comp_area - box_area) / max(comp_area, box_area)

    # Aspect ratio difference (normalized)
    aspect_ratio_diff = abs(comp_aspect_ratio - box_aspect_ratio)

    # Weighted score (lower is better)
    score = 0.8 * area_diff + 0.2 * aspect_ratio_diff
    return score


def postprocess_bboxes(bbox_predictions, width, height):
    # Scale the normalized bbox predictions to image dimensions
    processed_bboxes = []
    for bbox in bbox_predictions:
        xmin = bbox[0] * width
        ymin = bbox[1] * height
        xmax = bbox[2] * width
        ymax = bbox[3] * height
        processed_bboxes.append([xmin, ymin, xmax, ymax])

    return processed_bboxes

class ExistingLayoutGenerator():
    def __init__(self, table_path, device=None):
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load model weights
        self.layout_table = pd.read_csv(table_path)

    def predict_wi_area(self, num_bboxes, segment_list, width, height):
        # implement future
        pass
        
    def predict_wo_area(self, num_bboxes, width, height):
        # should we pass the result here
        layout_table_filtered = self.layout_table[self.layout_table['num_bboxes'] >= num_bboxes]
        # sample a layout from the filtered table
        bboxes_result = ast.literal_eval(layout_table_filtered.sample(n=1)['bboxes'].item())
        correspond_bboxes_result = bboxes_result[0:num_bboxes]
        # sort the box
        correspond_bboxes_result = sorted(
            bboxes_result, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True
        )
        # correspond_bboxes_result = random.sample(bboxes_result, num_bboxes)

        outputs = postprocess_bboxes(correspond_bboxes_result, width, height)
        return outputs
