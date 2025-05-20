import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
import math

# Function to scale normalized bounding boxes to pixel coordinates
def postprocess_bboxes(bboxes, width, height):
    processed = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        processed.append([int(x_min * width),
                          int(y_min * height),
                          int(min(x_max * width, width)),
                          int(min(y_max * height, height))])
    return processed

def compute_visible_ratios(bboxes):
    """
    For a given list of boxes (in generation order), compute the visible area ratio
    for each box (visible_area / total_area), assuming later boxes occlude earlier ones.
    """
    ratios = []
    polys = [shapely_box(*bbox) for bbox in bboxes]
    n = len(polys)
    for i in range(n):
        poly = polys[i]
        if i == n - 1:  # last box is fully visible
            ratios.append(1.0)
        else:
            occluders = polys[i+1:]
            union_occlusion = unary_union(occluders)
            intersection = poly.intersection(union_occlusion)
            occluded_area = intersection.area
            visible_area = poly.area - occluded_area
            ratio = visible_area / poly.area if poly.area > 0 else 0
            ratios.append(ratio)
    return ratios

def sample_box(min_area, max_area, aspect_range=(0.5, 2.0)):
    """
    Sample a box (in normalized coordinates) with an area in [min_area, max_area] and
    an aspect ratio in aspect_range.
    Returns [x_min, y_min, x_max, y_max] (all between 0 and 1), or None if failed.
    """
    # Sample a desired area uniformly
    area = random.uniform(min_area, max_area)
    # Sample an aspect ratio (width/height)
    aspect = random.uniform(aspect_range[0], aspect_range[1])
    # Compute width and height from area and aspect ratio:
    #   area = width * height, and width = aspect * height
    #   => height = sqrt(area / aspect), width = sqrt(area * aspect)
    h = math.sqrt(area / aspect)
    w = math.sqrt(area * aspect)
    # Ensure the box fits inside [0,1] by sampling a top-left coordinate.
    if w > 1 or h > 1:
        return None
    x_min = random.uniform(0, 1 - w)
    y_min = random.uniform(0, 1 - h)
    return [x_min, y_min, x_min + w, y_min + h]

class FineGrainedBoundingBoxLayoutGenerator():
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def generate(self, num_large, num_mid, num_small, width, height, 
                 # COCO area thresholds (normalized)
                 # For an image of area 1, small: <1024/1e6, medium: [1024/1e6, 9216/1e6), large: >=9216/1e6
                 # For a 1024x1024 image, total area=1.
                small_area_range=((32 * 16)/(640*480), (32 * 32)/(640*480)),          # ~[0, 0.0009766)
                mid_area_range=((32*32)/(640*480), (96 * 96)/(640*480)),   # ~[0.0009766, 0.0087891)
                large_area_range=((96 * 96)/(640*480), 0.5),          # large boxes: area >=0.0087891 and up to 0.5
                aspect_range=(0.5, 2.0),
                min_avg_display_area=0.8,
                min_single_display_area=0.5
                ):
        """
        Generate a layout of boxes in three groups:
          - Large boxes: number=num_large, area in large_area_range
          - Mid boxes:   number=num_mid, area in mid_area_range
          - Small boxes: number=num_small, area in small_area_range
        
        Boxes are generated sequentially (large first, then mid, then small).
        Overlap among boxes (regardless of group) is taken into account via visible area constraints.
        """
        total_boxes = num_large + num_mid + num_small
        max_layout_attempts = total_boxes * 100  # maximum attempts to generate the full layout
        layout_attempt = 0
        
        candidate_bboxes = []
        groups = [
            (num_large, large_area_range),
            (num_mid, mid_area_range),
            (num_small, small_area_range)
        ]
        
        while layout_attempt < max_layout_attempts:
            candidate_bboxes = []
            valid = True
            # For each group in order:
            for (num, area_range) in groups:
                for _ in range(num):
                    max_box_attempts = 100
                    box_found = False
                    for _ in range(max_box_attempts):
                        candidate_box = sample_box(area_range[0], area_range[1], aspect_range)
                        if candidate_box is None:
                            continue
                        temp_bboxes = candidate_bboxes + [candidate_box]
                        ratios = compute_visible_ratios(temp_bboxes)
                        avg_ratio = sum(ratios) / len(ratios)
                        if avg_ratio >= min_avg_display_area and all(r >= min_single_display_area for r in ratios):
                            candidate_bboxes.append(candidate_box)
                            box_found = True
                            break
                    if not box_found:
                        valid = False
                        break  # break out if a box in this group cannot be generated
                if not valid:
                    break
            if valid and len(candidate_bboxes) == total_boxes:
                return postprocess_bboxes(candidate_bboxes, width, height)
            layout_attempt += 1
        
        raise ValueError(f"After {max_layout_attempts} attempts, failed to generate a layout satisfying the constraints.")