from PIL import Image
import numpy as np

def generate_image_and_mask(data_manager, image_metadata, resize_mode, containSmallObjectMask=False, small_object_ratio=(36 * 36)/(640 * 480)):
    width = image_metadata["width"]
    height = image_metadata["height"]
    # Create a blank canvas and ground truth mask
    canvas = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    ground_truth_mask = Image.new("L", (width, height), 0)

    # This will collect segment IDs for small objects based on the bounding box area.
    small_object_ids = []

    # Add background if available
    if "background" in image_metadata:
        background_metadata = image_metadata["background"]["background_metadata"]
        background_image = data_manager.get_background_by_metadata(background_metadata)['image']
        canvas = paste_background(background_image, canvas)

    # Process objects
    if "objects" in image_metadata:
        for object_info in image_metadata["objects"]:
            object_metadata = object_info["object_metadata"]
            object_position = object_info["object_position"]  # either a bounding box or center point
            segment_id = object_info["segment_id"]

            # Use the bounding box area if available (i.e., when object_position has more than 2 elements).
            if len(object_position) != 2:
                bbox_area = (object_position[2] - object_position[0]) * (object_position[3] - object_position[1])
                if bbox_area / (width * height) <= small_object_ratio:
                    small_object_ids.append(segment_id)
            # Optionally, you might decide what to do if only a center point is provided.
            # For example, you could skip the small object check or assign a default small area.
            
            # Get the object image and its mask from the data manager.
            object_data_package = data_manager.get_object_by_metadata(object_metadata)
            image_obj, mask_obj = object_data_package['image'], object_data_package['mask']

            # If augmentation information is provided, apply the augmentation
            aug_params = object_info.get("augmentation", None)
            if aug_params is not None:
                image_obj, mask_obj = apply_augmentation(image_obj, mask_obj, aug_params)

            # Paste using the appropriate method based on how the object is defined.
            if len(object_position) == 2:
                paste_segment_wi_center_point(image_obj, mask_obj, canvas, ground_truth_mask, object_position, segment_id)
            else:
                paste_segment_wi_bbox(image_obj, mask_obj, canvas, ground_truth_mask, object_position, segment_id, resize_mode)

    # If a small object mask is desired, create it by keeping only the visible pixels for small objects.
    if containSmallObjectMask:
        ground_truth_mask_np = np.array(ground_truth_mask)
        # Retain only the pixels whose segment ID is in the small_object_ids list.
        small_object_mask_np = np.where(np.isin(ground_truth_mask_np, small_object_ids), ground_truth_mask_np, 0).astype(np.uint8)
        return canvas, ground_truth_mask, small_object_mask_np
    else:
        return canvas, ground_truth_mask

def paste_background(background, canvas):
    # Resize background if necessary
    if canvas.size != background.size:
        background = background.resize(canvas.size)
    return background

def normalize_mask(mask: Image) -> Image:
    """
    Convert a mask to mode 'L' and ensure its pixel values are in [0, 255].
    """
    if mask.mode != 'L':
        mask = mask.convert('L')
    mask_np = np.array(mask)
    if mask_np.max() <= 1:
        mask_np = (mask_np * 255).astype(np.uint8)
    return Image.fromarray(mask_np)

def paste_segment_wi_center_point(image: Image, mask: Image, canvas: Image,
                                  ground_truth_mask: Image, position: tuple, segment_id):
    # Ensure correct image modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    mask = normalize_mask(mask)

    # Get the bounding box of the nonzero (segmented) area
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("No segmented area found in the mask.")

    # Crop both the image and mask to the bounding box
    cropped_image = image.crop(bbox)
    cropped_mask = mask.crop(bbox)

    # Compute offset so that the segment's center aligns with the given position
    segment_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    offset = (int(position[0] - segment_center[0]), int(position[1] - segment_center[1]))

    # Build a segment ID mask using vectorized operations (avoid slow Python loops)
    cropped_mask_np = np.array(cropped_mask)
    segment_mask_np = np.where(cropped_mask_np > 0, segment_id, 0).astype(np.uint8)
    segment_id_mask = Image.fromarray(segment_mask_np)

    # Paste the cropped image and the segment ID mask onto their respective canvases
    canvas.paste(cropped_image, offset, cropped_mask)
    ground_truth_mask.paste(segment_id_mask, offset, cropped_mask)

def paste_segment_wi_bbox(image, mask, canvas,
                          ground_truth_mask, bbox, segment_id,
                          resize_mode):
    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    mask = normalize_mask(mask)

    # Get the bounding box of the segmented area to crop out unnecessary parts
    original_object_bbox = mask.getbbox()
    if original_object_bbox is None:
        raise ValueError("No segmented area found in the mask.")

    cropped_image = image.crop(original_object_bbox)
    cropped_mask = mask.crop(original_object_bbox)

    # Calculate the target region size based on the provided bounding box
    x_min, y_min, x_max, y_max = bbox
    bbox_width = int(round(x_max - x_min))
    bbox_height = int(round(y_max - y_min))

    if resize_mode == "full":
        # Stretch the image to fully fit the bounding box
        new_width, new_height = bbox_width, bbox_height
        offset = (int(round(x_min)), int(round(y_min)))
    elif resize_mode == "fit":
        # Maintain the aspect ratio of the cropped image
        orig_width, orig_height = cropped_image.size
        ratio = orig_width / orig_height
        # Compute the maximum size that fits within the bounding box while preserving the ratio
        if bbox_width / bbox_height > ratio:
            new_height = bbox_height
            new_width = int(round(bbox_height * ratio))
        else:
            new_width = bbox_width
            new_height = int(round(bbox_width / ratio))
        # Center the image within the bounding box
        offset = (int(round(x_min + (bbox_width - new_width) / 2)),
                  int(round(y_min + (bbox_height - new_height) / 2)))
    else:
        raise ValueError("resize_mode must be 'full' or 'fit'.")

    if new_width <= 0 or new_height <= 0:
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        # fixing now
    # Resize the cropped image and mask to the calculated dimensions
    resized_image = cropped_image.resize((new_width, new_height), resample=Image.BICUBIC)
    resized_mask = cropped_mask.resize((new_width, new_height), resample=Image.NEAREST)

    # Create a segment ID mask using vectorized operations
    resized_mask_np = np.array(resized_mask)
    segment_id_mask_np = np.where(resized_mask_np > 0, segment_id, 0).astype(np.uint8)
    segment_id_mask = Image.fromarray(segment_id_mask_np)

    # Paste the resized image and segment ID mask onto the canvas and ground truth mask
    canvas.paste(resized_image, offset, resized_mask)
    ground_truth_mask.paste(segment_id_mask, offset, resized_mask)

def apply_augmentation(image: Image, mask: Image, aug_params: dict):
    """
    Apply scaling, horizontal and vertical flips, and rotation to both the image and its mask.
    Scaling uses bicubic interpolation for the image and nearest for the mask;
    rotation is performed with expansion to preserve the full transformed segment.
    """
    scale = aug_params.get("scale", 1.0)
    flip_horizontal = aug_params.get("flip_horizontal", False)
    flip_vertical = aug_params.get("flip_vertical", False)
    rotate_angle = aug_params.get("rotate", 0)
    # Scaling: resize the image and mask if the scale factor is different from 1.0
    if scale != 1.0:
        new_size = (int(round(image.width * scale)), int(round(image.height * scale)))
        image = image.resize(new_size, resample=Image.BICUBIC)
        mask = mask.resize(new_size, resample=Image.NEAREST)

    # Horizontal flip: if the flag is True, perform a horizontal flip
    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip: if the flag is True, perform a vertical flip
    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation: if the rotation angle is non-zero, rotate the image and mask.
    if rotate_angle != 0:
        image = image.rotate(rotate_angle, expand=True, resample=Image.BICUBIC)
        mask = mask.rotate(rotate_angle, expand=True, resample=Image.NEAREST)

    return image, mask

