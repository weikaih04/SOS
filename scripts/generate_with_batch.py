import os
import sys

# Switch execution environment to parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.insert(0, project_root)

from datasets.object_datasets import *
from datasets.background_datasets import *
from datasets.data_manager import DataManager
from segmentation_generator.segmentataion_synthesizer import RandomCenterPointSegmentationSynthesizer, FineGrainedBoundingBoxSegmentationSynthesizer
import numpy as np
import json
from PIL import Image
import os
import random
import argparse
from tqdm import tqdm
import concurrent.futures


# Filter conditions
filtering_setting_0 = {
    "IsInstance": "non-filter",
    "Integrity": "non-filter",
    "QualityAndRegularity": "non-filter"
}
filtering_setting_1 = {
    "IsInstance": "filter",
    "Integrity": "non-filter",
    "QualityAndRegularity": "non-filter"
}
filtering_setting_2 = {
    "IsInstance": "non-filter",
    "Integrity": "filter",
    "QualityAndRegularity": "non-filter"
}
filtering_setting_3 = {
    "IsInstance": "non-filter",
    "Integrity": "non-filter",
    "QualityAndRegularity": "filter"
}
filtering_setting_4 = {
    "IsInstance": "filter",
    "Integrity": "filter",
    "QualityAndRegularity": "filter"
}


def reset_folders(paths):
    """Deletes and recreates specified directories to ensure a clean slate."""
    for path in paths:
        if os.path.exists(path):
            # Uncomment below to delete existing folders if needed:
            # shutil.rmtree(path)
            pass
        if not os.path.exists(path):
            print(f"Creating: {path}")
            os.makedirs(path, exist_ok=True)


def prepare_folders(image_save_path, mask_save_path, annotation_path):
    reset_folders([image_save_path, mask_save_path, annotation_path])


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

# without bg, for relightening and blending
def initialize_global_categories(annotation_path, filtering_setting, global_data_manager):
    """
    Initializes the global categories from the DataManager.
    This function runs in the main process so that the global categories
    are available when merging annotation JSONs.
    """
    categories = global_data_manager.unified_object_categories  
    categories_to_idx = {category: idx for idx, category in enumerate(categories)}
    idx_to_categories = {idx: category for idx, category in enumerate(categories)}
    global_categories = [{"id": idx, "name": category} for idx, category in enumerate(categories)]
    
    # Save category files for reference.
    json.dump(categories, open(os.path.join(annotation_path, "categories.json"), "w"), indent=4)
    json.dump(categories_to_idx, open(os.path.join(annotation_path, "categories_to_idx.json"), "w"), indent=4)
    json.dump(idx_to_categories, open(os.path.join(annotation_path, "idx_to_categories.json"), "w"), indent=4)
    
    return global_categories


def process_image_worker(start_idx, end_idx, worker_seed, filtering_setting, queue, image_save_path, mask_save_path, annotation_path, data_manager):
    """
    Worker function that processes a batch of images with a unique seed.
    For each image, the corresponding annotation is saved as a separate JSON file.
    """
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    rss = FineGrainedBoundingBoxSegmentationSynthesizer(data_manager, "./", random_seed=worker_seed)

    # Create a subfolder for separate annotations
    separate_annotation_path = os.path.join(annotation_path, "separate_annotations")
    if not (os.path.exists(separate_annotation_path) and os.path.isdir(separate_annotation_path)):
        os.makedirs(separate_annotation_path, exist_ok=True)

    for i in range(start_idx, end_idx):
        image_name = f"{i}.png"
        mask_name = f"{i}.png"
        image_path_full = os.path.join(image_save_path, image_name)
        mask_path_full = os.path.join(mask_save_path, mask_name)
        separate_annot_file = os.path.join(separate_annotation_path, f"{i}.json")

        # Skip processing if image, mask, and annotation all exist.
        if os.path.exists(image_path_full) and os.path.exists(mask_path_full) and os.path.exists(separate_annot_file):
            queue.put(1)
            continue

        obj_nums = np.random.randint(5, 20)
        data = rss.sampling_metadata(1024, 1024, obj_nums, hasBackground=False, dataAugmentation=False)
        res = rss.generate_with_unified_format(data, containRGBA=True, containCategory=True, containSmallObjectMask=False, resize_mode="fit")
        
        # Save image and mask.
        Image.fromarray(res['image_rgba']).save(image_path_full)
        Image.fromarray(res['coco_mask']).save(mask_path_full)

        # Build annotation for this image.
        annotation = {
            "segments_info": res["segments_info"],
            "file_name": mask_name,
            "image_id": i,
        }
        # Save the annotation as a separate JSON file.
        with open(separate_annot_file, "w") as f:
            json.dump(annotation, f, cls=NumpyEncoder, indent=4)

        queue.put(1)

# with bg
# def process_image_worker(start_idx, end_idx, worker_seed, filtering_setting, queue, image_save_path, mask_save_path, annotation_path, data_manager):
#     """
#     Worker function that processes a batch of images with a unique seed.
#     For each image, the corresponding annotation is saved as a separate JSON file.
#     """
#     random.seed(worker_seed)
#     np.random.seed(worker_seed)

#     rss = FineGrainedBoundingBoxSegmentationSynthesizer(data_manager, "./", random_seed=worker_seed)

#     # Create a subfolder for separate annotations
#     separate_annotation_path = os.path.join(annotation_path, "separate_annotations")
#     if not (os.path.exists(separate_annotation_path) and os.path.isdir(separate_annotation_path)):
#         os.makedirs(separate_annotation_path, exist_ok=True)

#     for i in range(start_idx, end_idx):
#         image_name = f"{i}.jpg"
#         mask_name = f"{i}.png"
#         image_path_full = os.path.join(image_save_path, image_name)
#         mask_path_full = os.path.join(mask_save_path, mask_name)
#         separate_annot_file = os.path.join(separate_annotation_path, f"{i}.json")

#         # Skip processing if image, mask, and annotation all exist.
#         if os.path.exists(image_path_full) and os.path.exists(mask_path_full) and os.path.exists(separate_annot_file):
#             queue.put(1)
#             continue

#         obj_nums = np.random.randint(5, 20)
#         data = rss.sampling_metadata(1024, 1024, obj_nums, hasBackground=True, dataAugmentation=False)
#         res = rss.generate_with_unified_format(data, containRGBA=True, containCategory=True, containSmallObjectMask=False, resize_mode="fit")
        
#         # Save image and mask.
#         res['image'].save(image_path_full)
#         Image.fromarray(res['coco_mask']).save(mask_path_full)

#         # Build annotation for this image.
#         annotation = {
#             "segments_info": res["segments_info"],
#             "file_name": mask_name,
#             "image_id": i,
#         }
#         # Save the annotation as a separate JSON file.
#         with open(separate_annot_file, "w") as f:
#             json.dump(annotation, f, cls=NumpyEncoder, indent=4)

#         queue.put(1)


def listener(queue, total):
    """Updates the tqdm progress bar as images are generated."""
    pbar = tqdm(total=total)
    while True:
        msg = queue.get()
        if msg == 'kill':
            break
        pbar.update(msg)
    pbar.close()




class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
        except ImportError:
            pass
        return super(NumpyEncoder, self).default(obj)

# Define process_file at the module level so it is pickleable
def process_file(annot_file):
    try:
        with open(annot_file, "r") as f:
            annot = json.load(f)
        # Build image info assuming a fixed size (1024x1024)
        image_info = {
            "file_name": f"{annot['image_id']}.jpg",
            "height": 1024,
            "width": 1024,
            "id": annot["image_id"]
        }
        return annot, image_info
    except Exception:
        return None

def merge_annotation_jsons(annotation_path, json_save_path, categories):
    """
    Merges separate annotation JSON files (stored in the "separate_annotations" subfolder)
    into a final COCO-format JSON. Also constructs the images list based on the annotation files.
    This version uses multiprocessing via a ProcessPoolExecutor with os.scandir for efficient directory listing.
    """
    separate_annotation_path = os.path.join(annotation_path, "separate_annotations")
    # Use os.scandir for efficient directory traversal
    json_files = [entry.path for entry in os.scandir(separate_annotation_path) 
                  if entry.is_file() and entry.name.endswith(".json")]
    
    annotations = []
    images = []
    
    # Adjust max_workers to a number more appropriate for your system (e.g., 4 to 8)
    max_workers = 100
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_file, json_files),
                            total=len(json_files),
                            desc="Merging annotations"))
        for res in results:
            if res is not None:
                annot, image_info = res
                annotations.append(annot)
                images.append(image_info)
    
    # Sort the lists based on their IDs
    images = sorted(images, key=lambda x: x["id"])
    annotations = sorted(annotations, key=lambda x: x["image_id"])
    
    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(json_save_path, "w") as f:
        json.dump(coco_json, f, cls=NumpyEncoder, indent=4)


def run_generation(args):
    """
    Runs the synthetic image generation process.
    Each worker saves its annotation into a separate JSON file.
    In the end, all these JSON files are merged into one final JSON.
    """
    image_save_path = args.image_save_path
    mask_save_path = args.mask_save_path
    annotation_path = args.annotation_path
    json_save_path = args.json_save_path

    prepare_folders(image_save_path, mask_save_path, annotation_path)

    # Set the filtering setting based on argument
    fs = str(args.filtering_setting)
    if fs.isdigit():
        fs = "filter_" + fs
    if fs == 'filter_0':
        filtering_setting = filtering_setting_0
    elif fs == 'filter_1':
        filtering_setting = filtering_setting_1
    elif fs == 'filter_2':
        filtering_setting = filtering_setting_2
    elif fs == 'filter_3':
        filtering_setting = filtering_setting_3
    else:
        filtering_setting = filtering_setting_4

    # Initialize global DataManager instance in the main process.
    global_data_manager = DataManager(available_object_datasets, available_background_datasets, filtering_setting)

    # Initialize global categories in the main process by passing global_data_manager.
    global_categories = initialize_global_categories(annotation_path, filtering_setting, global_data_manager)

    num_images = args.total_images
    num_workers = args.num_processes

    chunk_size = num_images // num_workers
    remainder = num_images % num_workers

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    listener_process = multiprocessing.Process(target=listener, args=(queue, num_images))
    listener_process.start()

    processes = []
    start_index = 0

    for i in range(num_workers):
        end_index = start_index + chunk_size
        if i < remainder:
            end_index += 1

        worker_seed = random.randint(0, int(1e6))

        def worker_task(start_idx=start_index, end_idx=end_index, worker_seed=worker_seed):
            process_image_worker(start_idx, end_idx, worker_seed, filtering_setting, queue, image_save_path, mask_save_path, annotation_path, data_manager=global_data_manager)

        p = multiprocessing.Process(target=worker_task)
        processes.append(p)
        p.start()
        start_index = end_index

    for p in processes:
        p.join()

    queue.put('kill')
    listener_process.join()

    # Merge individual annotation JSON files into the final COCO JSON.
    merge_annotation_jsons(annotation_path, json_save_path, global_categories)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Image Generation with Segmentation Annotations"
    )
    parser.add_argument('--num_processes', type=int, default=100,
                        help='Number of processes to use')
    parser.add_argument('--total_images', type=int, default=1000,
                        help='Total number of images to generate')
    parser.add_argument('--filtering_setting', type=str, default='filter_4',
                        help='Filtering setting to apply. Either "filter_1", "filter_2", "filter_3", "filter_4", or a digit 1-4.')
    parser.add_argument('--image_save_path', type=str,
                        default="/output/train",
                        help="Path to save images")
    parser.add_argument('--mask_save_path', type=str,
                        default="/output/panoptic_train",
                        help="Path to save masks")
    parser.add_argument('--annotation_path', type=str,
                        default="/output/annotations",
                        help="Path to save separate annotation JSON files")
    parser.add_argument('--json_save_path', type=str,
                        default="/output/annotations/panoptic_train.json",
                        help="Path to save the merged COCO panoptic JSON")
    args = parser.parse_args()

    run_generation(args)


if __name__ == "__main__":

    # using FC split data
    available_object_datasets = {
        "Synthetic": SyntheticDataset(
            dataset_path="/fc_10m", 
            synthetic_annotation_path="/fc_10m/gc_object_segments_metadata.json", 
            dataset_name="Synthetic",
            cache_path="./metadata_ovd_cache"
        ),
    }

    # using GC split data
    # available_object_datasets = {
    #     "Synthetic": SyntheticDataset(
    #         dataset_path="/gc_10m", 
    #         synthetic_annotation_path="/gc_10m/gc_object_segments_metadata.json", 
    #         dataset_name="Synthetic",
    #         cache_path="./metadata_ovd_cache"
    #     ),
    # }

    # mixing data from different source
    # available_object_datasets = {
    #     "Synthetic_fc": SyntheticDataset(
    #         dataset_path="/fc_10m", 
    #         synthetic_annotation_path="/fc_10m/gc_object_segments_metadata.json",
    #         dataset_name="Synthetic_fc",
    #         cache_path="./metadata_fc_cache"
    #     ),
    #     "Synthetic_gc": SyntheticDataset(
    #         dataset_path="/gc_10m", 
    #         synthetic_annotation_path="/gc_10m/gc_object_segments_metadata.json",
    #         dataset_name="Synthetic_gc",
    #         cache_path="./metadata_gc_cache"
    #     ),
    # }

    # By default, this loads from a placeholder dataset containing only a single image using `background`. 
    # To use `background` in BG20k, please set up the dataset path.
    available_background_datasets = {
        "BG20k": BG20KDataset("datasets/one_image_bg")
    #     "BG20k": BG20KDataset("/your/path/to/bg20k_dataset")

    }

    main()
