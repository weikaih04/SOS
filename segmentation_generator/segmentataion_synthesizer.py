import numpy as np
from .generate_image_and_mask import generate_image_and_mask
from segment_layout.existing_layout_generator import ExistingLayoutGenerator
from segment_layout.fine_grained_bbox_layout_generator import FineGrainedBoundingBoxLayoutGenerator
from PIL import Image

def generate_random_color_and_id(rng):
    # Generate random values for R, G, and B between 0 and 255
    R = rng.integers(0, 256)
    G = rng.integers(0, 256)
    B = rng.integers(0, 256)

    # Calculate the color id based on the formula provided
    color_id = R + G * 256 + B * (256**2)

    # Return the color and the id
    color = (R, G, B)
    return color, color_id


class BaseSegmentationSynthesizer():
    synthesize_method = None
    def __init__(self, data_manager, save_path, random_seed) -> None:
        if random_seed is None:
            random_seed = 42 # use the default seed
        self.rng = np.random.default_rng(random_seed)
        self.data_manager = data_manager
        self.save_path = save_path
        
    def random_position(self, width, height):
        x = self.rng.integers(0, width)
        y = self.rng.integers(0, height)
        return (x, y)

    def random_augmentation(self):
        # Randomly select a scale factor between 0.5 and 1.5,
        # a boolean flag for horizontal flip (50% chance),
        # a boolean flag for vertical flip (50% chance),
        # and a rotation angle between -180 and 180 degrees.
        scale = self.rng.uniform(0.5, 1.5)
        flip_horizontal = bool(self.rng.integers(0, 2))
        flip_vertical = bool(self.rng.integers(0, 2))
        rotate = int(self.rng.integers(-180, 180))
        return {"scale": scale, "flip_horizontal": flip_horizontal, "flip_vertical": flip_vertical, "rotate": rotate}



    def category_to_id(self, category):  
        if category.startswith("COCO2017"):
            return int(category.split("_")[-1])
        if category in self.data_manager.unified_object_categories_to_idx:
            return self.data_manager.unified_object_categories_to_idx[category]
        else:
            raise ValueError(f"Category {category} not found in the unified object categories.")
    
    def generate(self, image_metadata, resize_mode, containSmallObjectMask=False):
        # start_time = time.time()
        if containSmallObjectMask:
            image, mask, small_object_mask_np = generate_image_and_mask(self.data_manager, image_metadata, resize_mode=resize_mode, containSmallObjectMask=True)
            output = {"image_metadata": image_metadata, "image": image, "mask": mask, "small_object_mask_np": small_object_mask_np}
        else:
            image, mask = generate_image_and_mask(self.data_manager, image_metadata, resize_mode=resize_mode, containSmallObjectMask=False)
            output = {"image_metadata": image_metadata, "image": image, "mask": mask}
        
        # elapsed_time = time.time() - start_time  # compute elapsed time
        # print(f"generate_image_and_mask took {elapsed_time:.6f} seconds")
        
        return output


    def generate_with_coco_panoptic_format(self, image_metadata, resize_mode="fit", containRGBA=False, containBbox=True, containCategory=True, containSmallObjectMask=False):
        output = self.generate(image_metadata, resize_mode, containSmallObjectMask)
        image, mask, metadata = output["image"], output["mask"], output["image_metadata"]
        
        segment_id_to_color = {}
        used_colors = set()
        mask = np.array(mask)
        height, width = mask.shape

        coco_mask = np.zeros((height, width, 3), dtype=np.uint8)
        segments_info = []

        for object_info in metadata["objects"]:
            segment_id = object_info["segment_id"]
            if segment_id != 0:
                while True:
                    unique_color, unique_id = generate_random_color_and_id(self.rng)
                    if unique_color not in used_colors:
                        used_colors.add(unique_color)  # Mark this color as used
                        break

                if segment_id not in segment_id_to_color:
                    segment_id_to_color[segment_id] = (unique_color, unique_id)
                coco_mask[mask == segment_id] = segment_id_to_color[segment_id][0]

                if containBbox:
                    rows, cols = np.where(mask == segment_id)
                    if rows.size and cols.size:
                        x_min, y_min = int(cols.min()), int(rows.min())
                        x_max, y_max = int(cols.max()), int(rows.max())
                        # COCO uses [x, y, width, height] format.
                        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                    else:
                        bbox = [0, 0, 0, 0]
                else:
                    bbox = [0, 0, 0, 0]

                # iscrowd, bbox, area are not used in this case, so we set them to dummy values
                segments_info.append({
                    "id": unique_id,
                    "category_id": self.category_to_id(object_info['object_metadata']["category"]) if containCategory else 1,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "area": 1,
                })

        output["coco_mask"] = coco_mask
        output["segments_info"] = segments_info
        if containRGBA:
            alpha_channel = (mask > 0).astype(np.uint8) * 255
            image_rgba = np.dstack((image, alpha_channel))
            output["image_rgba"] = image_rgba
        if containSmallObjectMask:
            alpha_channel_small_obj = (output['small_object_mask_np'] > 0).astype(np.uint8) * 255
            image_rgba_small_obj = np.dstack((image, alpha_channel_small_obj))
            output['image_rgba_small_object'] = image_rgba_small_obj
        return output


    def generate_with_unified_format(self, image_metadata, resize_mode="fit", containRGBA=False, containBbox=True, containCategory=True, containSmallObjectMask=False):
        output = self.generate(image_metadata, resize_mode, containSmallObjectMask)
        image, mask, metadata = output["image"], output["mask"], output["image_metadata"]
        
        segment_id_to_color = {}
        used_colors = set()
        mask = np.array(mask)
        height, width = mask.shape

        coco_mask = np.zeros((height, width, 3), dtype=np.uint8)
        segments_info = []

        for object_info in metadata["objects"]:
            segment_id = object_info["segment_id"]
            if segment_id != 0:
                while True:
                    unique_color, unique_id = generate_random_color_and_id(self.rng)
                    if unique_color not in used_colors:
                        used_colors.add(unique_color)  # Mark this color as used
                        break

                if segment_id not in segment_id_to_color:
                    segment_id_to_color[segment_id] = (unique_color, unique_id)
                coco_mask[mask == segment_id] = segment_id_to_color[segment_id][0]

                if containBbox:
                    rows, cols = np.where(mask == segment_id)
                    if rows.size and cols.size:
                        x_min, y_min = int(cols.min()), int(rows.min())
                        x_max, y_max = int(cols.max()), int(rows.max())
                        # COCO uses [x, y, width, height] format.
                        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                    else:
                        bbox = [0, 0, 0, 0]
                else:
                    bbox = [0, 0, 0, 0]

                # iscrowd, bbox, area are not used in this case, so we set them to dummy values
                segments_info.append({
                    "id": unique_id,
                    "category_id": self.category_to_id(object_info['object_metadata']["category"]) if containCategory else 1,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "category": object_info['object_metadata']["category"],
                    "sub_category": object_info['object_metadata']["sub_category"],
                    "description": object_info['object_metadata']["description"],
                    "features": object_info['object_metadata']["features"],
                    "short_phrase": object_info['object_metadata']["short_phrase"],
                    "area": 1,
                })

        output["coco_mask"] = coco_mask
        output["segments_info"] = segments_info
        if containRGBA:
            alpha_channel = (mask > 0).astype(np.uint8) * 255
            image_rgba = np.dstack((image, alpha_channel))
            output["image_rgba"] = image_rgba
        if containSmallObjectMask:
            alpha_channel_small_obj = (output['small_object_mask_np'] > 0).astype(np.uint8) * 255
            image_rgba_small_obj = np.dstack((image, alpha_channel_small_obj))
            output['image_rgba_small_object'] = image_rgba_small_obj
        return output

class RandomCenterPointSegmentationSynthesizer(BaseSegmentationSynthesizer):
    synthesize_method = "random"
    def __init__(self, data_manager, save_path, random_seed=None):
        super().__init__(data_manager, save_path, random_seed)

    def sampling_metadata(self, width, height, number_of_objects, hasBackground=False, dataAugmentation=False):  # -> dict:
        image_metadata = {}
        image_metadata["width"] = width
        image_metadata["height"] = height
        image_metadata["number_of_objects"] = number_of_objects
        image_metadata["synthesize_method"] = self.synthesize_method

        if hasBackground:
            background_metadata = self.data_manager.get_random_background_metadata(self.rng)
            image_metadata["background"] = {"background_metadata": background_metadata}
        
        curr_segment_id = 1
        objects = []
        for _ in range(number_of_objects):
            # Randomly select a segmentation from the data manager
            object_metadata = self.data_manager.get_random_object_metadata(self.rng)
            object_position = self.random_position(width, height)
            if dataAugmentation:
                augmentation = self.random_augmentation()
                objects.append({
                    "object_metadata": object_metadata, 
                    "object_position": object_position, 
                    "segment_id": curr_segment_id,
                    "augmentation": augmentation
                })
            else:
                objects.append({
                    "object_metadata": object_metadata, 
                    "object_position": object_position, 
                    "segment_id": curr_segment_id
                })
            curr_segment_id += 1
        image_metadata["objects"] = objects
        return image_metadata

class FineGrainedBoundingBoxSegmentationSynthesizer(BaseSegmentationSynthesizer):
    synthesize_method = "fine_grained_bbox"
    def __init__(self, data_manager, save_path, random_seed=None):
        super().__init__(data_manager, save_path, random_seed)
        self.fine_grained_bbox_layout_generator = FineGrainedBoundingBoxLayoutGenerator()

    def sampling_metadata(self, width, height, number_of_objects, hasBackground=False, dataAugmentation=False, considerArea=False):  # -> dict:
        image_metadata = {}
        image_metadata["width"] = width
        image_metadata["height"] = height
        image_metadata["number_of_objects"] = number_of_objects
        image_metadata["synthesize_method"] = self.synthesize_method

        if hasBackground:
            background_metadata = self.data_manager.get_random_background_metadata(self.rng)
            image_metadata["background"] = {"background_metadata": background_metadata}
        
        curr_segment_id = 1
        objects = []
        # sort based on the original size
        for _ in range(number_of_objects):
            # Randomly select a segmentation from the data manager
            object_metadata = self.data_manager.get_random_object_metadata(self.rng)
            if considerArea:
                # add area
                path = object_metadata['image_path']
                with Image.open(path) as img:
                    img_np = np.array(img)
                    area = np.sum(img_np[:, :, 3] > 0)
                    object_metadata['area'] = area

            if dataAugmentation:
                augmentation = self.random_augmentation()
                objects.append({
                    "object_metadata": object_metadata, 
                    "segment_id": curr_segment_id,
                    "augmentation": augmentation
                })
            else:
                objects.append({
                    "object_metadata": object_metadata, 
                    "segment_id": curr_segment_id
                })
            curr_segment_id += 1

        num_bboxes = len(objects)
        # based on the COCO website
        num_large = int(round(num_bboxes * 0.24))
        num_mid = int(round(num_bboxes * 0.34))
        num_small = num_bboxes - num_large - num_mid
        object_positions = self.fine_grained_bbox_layout_generator.generate(num_large=num_large, num_mid=num_mid, 
                                                                            num_small=num_small, width=width, height=height)
        if considerArea:
            bbox_with_area = []
            for bbox in object_positions:
                # Calculate width and height from [x_min, y_min, x_max, y_max]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_area = width * height
                bbox_with_area.append({"bbox": bbox, "area": bbox_area})
                
            # Sort objects based on the computed object area (descending order)
            objects_sorted = sorted(objects, key=lambda x: x['object_metadata']['area'], reverse=True)
            # Sort bounding boxes based on their computed area (descending order)
            bboxes_sorted = sorted(bbox_with_area, key=lambda x: x['area'], reverse=True)
            
            # Assign the sorted bounding boxes to the sorted objects
            for idx, obj in enumerate(objects_sorted):
                obj["object_position"] = bboxes_sorted[idx]["bbox"]
            image_metadata["objects"] = objects_sorted
            # print the match results
            for obj in objects_sorted:
                print(f"Object: {obj['object_metadata']['image_path']}, Area: {obj['object_metadata']['area']}")
            for bbox in bboxes_sorted:
                print(f"bbox: {bbox['bbox']}, Area: {bbox['area']}")
            print("in")
        else:
            for idx, obj in enumerate(objects):
                obj["object_position"] = object_positions[idx]
            image_metadata["objects"] = objects

        return image_metadata


class FineGrainedBoundingBoxSingleCategorySegmentationSynthesizer(BaseSegmentationSynthesizer):
    synthesize_method = "fine_grained_bbox"
    def __init__(self, data_manager, save_path, random_seed=None):
        super().__init__(data_manager, save_path, random_seed)
        self.fine_grained_bbox_layout_generator = FineGrainedBoundingBoxLayoutGenerator()

    def sampling_metadata(
        self,
        width,
        height,
        number_of_objects,
        hasBackground=False,
        dataAugmentation=False,
        considerArea=False
    ):
        image_metadata = {
            "width": width,
            "height": height,
            "number_of_objects": number_of_objects,
            "synthesize_method": self.synthesize_method
        }

        if hasBackground:
            bg_meta = self.data_manager.get_random_background_metadata(self.rng)
            image_metadata["background"] = {"background_metadata": bg_meta}

        # First object: random to determine category
        first_meta = self.data_manager.get_random_object_metadata(self.rng)
        category = first_meta.get('category')
        if category is None:
            raise KeyError("First sampled object metadata has no 'category' field.")

        objects = []
        curr_id = 1
        # Process first object
        if considerArea:
            path = first_meta['image_path']
            with Image.open(path) as img:
                arr = np.array(img)
                first_meta['area'] = int(np.sum(arr[:, :, 3] > 0))
        entry = {"object_metadata": first_meta, "segment_id": curr_id}
        if dataAugmentation:
            entry["augmentation"] = self.random_augmentation()
        objects.append(entry)
        curr_id += 1

        # Sample remaining objects of the same category
        for _ in range(number_of_objects - 1):
            obj_meta = self.data_manager.get_random_object_metadata_by_category(self.rng, category)
            if considerArea:
                path = obj_meta['image_path']
                with Image.open(path) as img:
                    arr = np.array(img)
                    obj_meta['area'] = int(np.sum(arr[:, :, 3] > 0))
            entry = {"object_metadata": obj_meta, "segment_id": curr_id}
            if dataAugmentation:
                entry["augmentation"] = self.random_augmentation()
            objects.append(entry)
            curr_id += 1

        # Compute bbox size splits per COCO proportions
        n = len(objects)
        n_large = int(round(n * 0.24))
        n_mid = int(round(n * 0.34))
        n_small = n - n_large - n_mid
        bboxes = self.fine_grained_bbox_layout_generator.generate(
            num_large=n_large,
            num_mid=n_mid,
            num_small=n_small,
            width=width,
            height=height
        )

        if considerArea:
            boxes_with_area = [
                {"bbox": box, "area": (box[2] - box[0]) * (box[3] - box[1])}
                for box in bboxes
            ]
            objects_sorted = sorted(objects, key=lambda o: o['object_metadata']['area'], reverse=True)
            boxes_sorted = sorted(boxes_with_area, key=lambda b: b['area'], reverse=True)
            for obj, box in zip(objects_sorted, boxes_sorted):
                obj['object_position'] = box['bbox']
            image_metadata['objects'] = objects_sorted
        else:
            for obj, box in zip(objects, bboxes):
                obj['object_position'] = box
            image_metadata['objects'] = objects

        return image_metadata




class ExistingLayoutSegmentationSynthesizer(BaseSegmentationSynthesizer):
    synthesize_method = "existing_layout"
    def __init__(self, data_manager, save_path, table_path, random_seed = None):
        super().__init__(data_manager, save_path, random_seed)
        self.existing_layout_generator = ExistingLayoutGenerator(table_path=table_path)
        
    def sampling_metadata(self, width, height, number_of_objects, hasBackground=False, dataAugmentation=None, considerArea=False):# -> dict:
        image_metadata = {}
        image_metadata["width"] = width
        image_metadata["height"] = height
        image_metadata["number_of_objects"] = number_of_objects
        image_metadata["synthesize_method"] = self.synthesize_method
        # add background image background 

        if hasBackground:
            background_metadata = self.data_manager.get_random_background_metadata(self.rng)
            image_metadata["background"] = {"background_metadata": background_metadata}
        
        curr_segment_id = 1
        objects = []
        # sort based on the original size
        for _ in range(number_of_objects):
            # Randomly select a segmentation from the data manager
            object_metadata = self.data_manager.get_random_object_metadata(self.rng)
            if dataAugmentation:
                augmentation = self.random_augmentation()
                objects.append({
                    "object_metadata": object_metadata, 
                    "segment_id": curr_segment_id,
                    "augmentation": augmentation
                })
            else:
                objects.append({
                    "object_metadata": object_metadata, 
                    "segment_id": curr_segment_id
                })
            curr_segment_id += 1
            
        segment_path_list = [obj['object_metadata']['image_path'] for obj in objects]
        if considerArea:
            # havn't implemented the area matching algorithm
            # object_areas = [obj['object_metadata']['area'] for obj in objects]
            object_areas = segment_path_list
            object_positions = self.existing_layout_generator.predict_wi_area(num_bboxes=len(objects), segment_path_list=segment_path_list, areas=object_areas, width=width, height=height)
        else:
            object_positions = self.existing_layout_generator.predict_wo_area(num_bboxes=len(objects), width=width, height=height)

        for idx, obj in enumerate(objects):
            obj["object_position"] = object_positions[idx]
        image_metadata["objects"] = objects

        return image_metadata


class FineGrainedBoundingBoxSegmentationSynthesizer(BaseSegmentationSynthesizer):
    synthesize_method = "fine_grained_bbox"
    def __init__(self, data_manager, save_path, random_seed=None):
        super().__init__(data_manager, save_path, random_seed)
        self.fine_grained_bbox_layout_generator = FineGrainedBoundingBoxLayoutGenerator()

    def sampling_metadata(self, width, height, number_of_objects, hasBackground=False, dataAugmentation=False, considerArea=False):  # -> dict:
        image_metadata = {}
        image_metadata["width"] = width
        image_metadata["height"] = height
        image_metadata["number_of_objects"] = number_of_objects
        image_metadata["synthesize_method"] = self.synthesize_method

        if hasBackground:
            background_metadata = self.data_manager.get_random_background_metadata(self.rng)
            image_metadata["background"] = {"background_metadata": background_metadata}
        
        curr_segment_id = 1
        objects = []
        # sort based on the original size
        for _ in range(number_of_objects):
            # Randomly select a segmentation from the data manager
            object_metadata = self.data_manager.get_random_object_metadata(self.rng)
            if considerArea:
                # add area
                path = object_metadata['image_path']
                with Image.open(path) as img:
                    img_np = np.array(img)
                    area = np.sum(img_np[:, :, 3] > 0)
                    object_metadata['area'] = area

            if dataAugmentation:
                augmentation = self.random_augmentation()
                objects.append({
                    "object_metadata": object_metadata, 
                    "segment_id": curr_segment_id,
                    "augmentation": augmentation
                })
            else:
                objects.append({
                    "object_metadata": object_metadata, 
                    "segment_id": curr_segment_id
                })
            curr_segment_id += 1

        num_bboxes = len(objects)
        # based on the COCO website
        num_large = int(round(num_bboxes * 0.24))
        num_mid = int(round(num_bboxes * 0.34))
        num_small = num_bboxes - num_large - num_mid
        object_positions = self.fine_grained_bbox_layout_generator.generate(num_large=num_large, num_mid=num_mid, 
                                                                            num_small=num_small, width=width, height=height)
        if considerArea:
            bbox_with_area = []
            for bbox in object_positions:
                # Calculate width and height from [x_min, y_min, x_max, y_max]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_area = width * height
                bbox_with_area.append({"bbox": bbox, "area": bbox_area})
                
            # Sort objects based on the computed object area (descending order)
            objects_sorted = sorted(objects, key=lambda x: x['object_metadata']['area'], reverse=True)
            # Sort bounding boxes based on their computed area (descending order)
            bboxes_sorted = sorted(bbox_with_area, key=lambda x: x['area'], reverse=True)
            
            # Assign the sorted bounding boxes to the sorted objects
            for idx, obj in enumerate(objects_sorted):
                obj["object_position"] = bboxes_sorted[idx]["bbox"]
            image_metadata["objects"] = objects_sorted
            # print the match results
            for obj in objects_sorted:
                print(f"Object: {obj['object_metadata']['image_path']}, Area: {obj['object_metadata']['area']}")
            for bbox in bboxes_sorted:
                print(f"bbox: {bbox['bbox']}, Area: {bbox['area']}")
        else:
            for idx, obj in enumerate(objects):
                obj["object_position"] = object_positions[idx]
            image_metadata["objects"] = objects

        return image_metadata