from .base_datasets import BaseObjectDataset
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json
import re
from shutil import copyfile
from PIL import Image
import PIL
from .coco_categories import COCO_CATEGORIES
def build_category_index(categories):
    return {category['name'].replace("_", "").replace(" ", ""): category['id'] for category in categories}

COCO_CATEGORY_INDEX = build_category_index(COCO_CATEGORIES)


def getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation):
    # getting the first value of the filtering_annotations as schema


    if universial_idx in filtering_annotations:
        return filtering_annotations[universial_idx]
    else:
        return default_annotation

class ADE20KDataset(BaseObjectDataset):
    def __init__(self, dataset_path, filtering_annotations_path):
        dataset_name = "ADE20K" 
        data_type = "panoptic"
        super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
        # get filtering annotation
        if self.filtering_annotations_path is not None:
            filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
            default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
            default_annotation = {key: False for key in default_annotation.keys()}
        else:
            filtering_annotations = None
        
        # get metadata_table
        index_counter = 0
        rows = []
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if ".png" in file_name:
                        image_path = os.path.join(self.dataset_path, folder_name, file_name)
                        universial_idx = f"{dataset_name}_{folder_name}_{file_name}"     
                        rows.append({
                            'universal_idx': universial_idx,
                            'idx_for_curr_dataset': index_counter,
                            'category': folder_name,
                            'image_path': image_path,
                            'mask_path': image_path,  # Assuming mask_path is the same as image_path
                            'dataset_name': dataset_name,
                            'data_type': data_type,
                            'filtering_annotation': getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation) if filtering_annotations is not None else None
                        })
                        index_counter += 1
        self.metadata_table = pd.DataFrame(rows)
        self.categories = set(self.metadata_table['category'])

class VOC2012Dataset(BaseObjectDataset):
    def __init__(self, dataset_path, filtering_annotations_path):
        dataset_name = "VOC2012" 
        data_type = "panoptic"
        super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
        # get filtering annotation
        if self.filtering_annotations_path is not None:
            filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
            default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
            default_annotation = {key: False for key in default_annotation.keys()}
        else:
            filtering_annotations = None
        
        # get metadata_table
        index_counter = 0
        rows = []
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if ".png" in file_name:
                        image_path = os.path.join(self.dataset_path, folder_name, file_name)
                        universial_idx = f"{dataset_name}_{folder_name}_{file_name}"  
                        rows.append({
                            'universal_idx': universial_idx,
                            'idx_for_curr_dataset': index_counter,
                            'category': folder_name,
                            'image_path': image_path,
                            'mask_path': image_path,  # Assuming mask_path is the same as image_path
                            'dataset_name': dataset_name,
                            'data_type': data_type,
                            'filtering_annotation': getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation) if filtering_annotations is not None else None
                        })
                        index_counter += 1
        self.metadata_table = pd.DataFrame(rows)
        self.categories = set(self.metadata_table['category'])



# First, build an index mapping cleaned category names to the category data.
# Now define the lookup function that uses the pre-built index.

class COCO2017Dataset(BaseObjectDataset):
    def _load_coco_id(self, folder_name: str) -> dict:
        tokens = folder_name.strip().split()
        if not tokens:
            raise ValueError("Empty folder name provided.")
        # Remove trailing underscore and digits from the first token.
        cleaned_folder_name = re.sub(r'\d+', '', tokens[0])
        cleaned_folder_name = cleaned_folder_name.replace("_", "")
        try:
            return f"COCO2017_{cleaned_folder_name}_{COCO_CATEGORY_INDEX[cleaned_folder_name]}"
        except KeyError:
            raise ValueError(f"Could not find COCO category for folder name '{folder_name}, cleaned to '{cleaned_folder_name}'.")
     
    def __init__(self, dataset_path, filtering_annotations_path=None, available_coco_image_path=None, coco_segment_to_image_path=None, dataset_name="COCO2017"):
        data_type = "panoptic"
        super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
        # get filtering annotation
        if self.filtering_annotations_path is not None:
            filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
            default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
            default_annotation = {key: False for key in default_annotation.keys()}
        else:
            filtering_annotations = None

        # add coco image limit, limit the source image that each segments coming from
        if available_coco_image_path is not None:
            available_coco_image_list = json.load(open(available_coco_image_path, "r"))
            print(f"Only consider {len(available_coco_image_list)} images from COCO2017")

        # get metadata_table
        index_counter = 0
        rows = []
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if ".png" in file_name:

                        # check if segment are not in the available_coco_image_list, then remove it from segments pool
                        if available_coco_image_path is not None:
                            correspond_image_name = file_name.split("_")[0] + ".jpg"
                            if correspond_image_name not in available_coco_image_list:
                                continue

                        image_path = os.path.join(self.dataset_path, folder_name, file_name)
                        universial_idx = f"{dataset_name}_{folder_name}_{file_name}"
                        rows.append({
                            'universal_idx': universial_idx,
                            'idx_for_curr_dataset': index_counter,
                            'category': self._load_coco_id(folder_name),
                            'image_path': image_path,
                            'mask_path': image_path,  # Assuming mask_path is the same as image_path
                            'dataset_name': dataset_name,
                            'data_type': data_type,
                            'filtering_annotation': getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation) if filtering_annotations is not None else None
                        })
                        index_counter += 1
        self.metadata_table = pd.DataFrame(rows)
        self.categories = set(self.metadata_table['category'])


class COCO2017FullDataset(COCO2017Dataset):
    def __init__(self, dataset_path, filtering_annotations_path=None, available_coco_image_path=None, coco_segment_to_image_path=None):
        super().__init__(dataset_path, filtering_annotations_path, available_coco_image_path, coco_segment_to_image_path, dataset_name="COCO_Full")
        


# class SyntheticDataset(BaseObjectDataset):     
#     def __init__(self, dataset_path, filtering_annotations_path=None, synthetic_annotation_path=None, dataset_name="SyntheticDatasetPlaceHolder"):
#         data_type = "panoptic"
#         super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
#         # get filtering annotation
#         if self.filtering_annotations_path is not None:
#             filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
#             default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
#             default_annotation = {key: False for key in default_annotation.keys()}
#         else:
#             filtering_annotations = None

#         if synthetic_annotation_path is not None:
#             synthetic_annotation = json.load(open(synthetic_annotation_path, "r"))

#         # get metadata_table
#         index_counter = 0
#         rows = []

#         # Wrap the first layer iteration with tqdm for progress display
#         for folder_name in tqdm(os.listdir(self.dataset_path), desc="Processing top-level directories"):
#             folder_path = os.path.join(self.dataset_path, folder_name)
#             if os.path.isdir(folder_path):
#                 for subfolder_name in os.listdir(folder_path):
#                     subfolder_path = os.path.join(folder_path, subfolder_name)
#                     if os.path.isdir(subfolder_path):
#                         for file_name in os.listdir(subfolder_path):
#                             if ".png" in file_name:
#                                 # check if segment are not in the available_coco_image_list, then remove it from segments pool
#                                 image_path = os.path.join(self.dataset_path, folder_name, subfolder_name, file_name)
#                                 image_subfolder_id = file_name.replace('.png', '').split("_")[-1]

#                                 universial_idx = f"{dataset_name}_{folder_name}_{subfolder_name}_{image_subfolder_id}"
#                                 query_idx = f"{folder_name}_{subfolder_name}"

#                                 if query_idx in synthetic_annotation:
#                                     description = synthetic_annotation[query_idx]["description"]
#                                     short_phrase = synthetic_annotation[query_idx]["short_phrase"]
#                                     features = synthetic_annotation[query_idx]["features"]
#                                     rows.append({
#                                         'universal_idx': universial_idx,
#                                         'query_idx': query_idx,
#                                         'idx_for_curr_dataset': index_counter,
#                                         'image_path': image_path,
#                                         'mask_path': image_path,  # Assuming mask_path is the same as image_path
#                                         'dataset_name': dataset_name,
#                                         'data_type': data_type,
#                                         'filtering_annotation': getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation) if filtering_annotations is not None else None,
#                                         'category': folder_name,
#                                         'sub_category': subfolder_name,
#                                         'description': description,
#                                         'short_phrase': short_phrase,
#                                         'features': features
#                                     })
#                                     index_counter += 1
#         self.metadata_table = pd.DataFrame(rows)
#         self.categories = set(self.metadata_table['category'])


class SyntheticDataset(BaseObjectDataset):     
    def __init__(self, dataset_path, filtering_annotations_path=None, synthetic_annotation_path=None, dataset_name="SyntheticDatasetPlaceHolder", cache_path="metadata_cache.pkl"):
        data_type = "panoptic"
        super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
        
        # Load filtering annotations if provided
        if self.filtering_annotations_path is not None:
            filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
            default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
            default_annotation = {key: False for key in default_annotation.keys()}
        else:
            filtering_annotations = None
            default_annotation = None

        if synthetic_annotation_path is not None:
            synthetic_annotation = json.load(open(synthetic_annotation_path, "r"))
        else:
            synthetic_annotation = {}

        # Check if cache exists:
        if os.path.exists(cache_path):
            print(f"Loading metadata from cache: {cache_path}")
            with open(cache_path, "rb") as fp:
                self.metadata_table = pickle.load(fp)
        else:
            # If cache does not exist, create the metadata table
            self.metadata_table = self._build_metadata_table(synthetic_annotation, filtering_annotations, default_annotation, dataset_name)
            # Save to cache for future runs
            with open(cache_path, "wb") as fp:
                pickle.dump(self.metadata_table, fp)
            print(f"Metadata cache saved to: {cache_path}")

        # Create a set of categories for later use
        self.categories = set(self.metadata_table['category'])
    
    def _process_folder(self, folder_name, synthetic_annotation, filtering_annotations, default_annotation, dataset_name):
        """
        Process one top-level directory and return a list of rows (dicts) for that folder.
        """
        index_counter = 0  # each folder can have an independent counter, we'll fix indices after merging if needed
        rows = []
        folder_path = os.path.join(self.dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    for file_name in os.listdir(subfolder_path):
                        if file_name.endswith(".png"):
                            image_path = os.path.join(self.dataset_path, folder_name, subfolder_name, file_name)
                            image_subfolder_id = file_name.replace('.png', '').split("_")[-1]

                            universial_idx = f"{dataset_name}_{folder_name}_{subfolder_name}_{image_subfolder_id}"
                            query_idx = f"{folder_name}_{subfolder_name}"

                            if query_idx in synthetic_annotation:
                                description = synthetic_annotation[query_idx]["description"]
                                short_phrase = synthetic_annotation[query_idx]["short_phrase"]
                                features = synthetic_annotation[query_idx]["features"]
                                # Get filtering annotation if exists
                                filtering_ann = (getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation)
                                                   if filtering_annotations is not None else None)
                                rows.append({
                                    'universal_idx': universial_idx,
                                    'query_idx': query_idx,
                                    'image_path': image_path,
                                    'mask_path': image_path,  # assuming mask_path is the same as image_path
                                    'dataset_name': dataset_name,
                                    'data_type': "panoptic",
                                    'filtering_annotation': filtering_ann,
                                    'category': folder_name,
                                    'sub_category': subfolder_name,
                                    'description': description,
                                    'short_phrase': short_phrase,
                                    'features': features
                                })
                                index_counter += 1
        return rows

    def _build_metadata_table(self, synthetic_annotation, filtering_annotations, default_annotation, dataset_name):
        """
        Build the metadata table from the dataset_path using multithreading.
        """
        all_rows = []
        
        # Get the list of top-level directories
        folders = [folder for folder in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, folder))]
        
        # Use a ThreadPoolExecutor to process folders concurrently
        with ThreadPoolExecutor(max_workers=600) as executor:
            # Submit jobs for each top-level folder
            future_to_folder = {
                executor.submit(self._process_folder, folder, synthetic_annotation, filtering_annotations, default_annotation, dataset_name): folder
                for folder in folders
            }
            # Use tqdm to monitor progress
            for future in tqdm(as_completed(future_to_folder), total=len(future_to_folder), desc="Processing folders"):
                folder = future_to_folder[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                except Exception as exc:
                    print(f"Folder {folder} generated an exception: {exc}")
        
        # Optionally, if you want sequential indexing across all rows:
        for idx, row in enumerate(all_rows):
            row["idx_for_curr_dataset"] = idx
        
        return pd.DataFrame(all_rows)


# class SyntheticDataset(BaseObjectDataset):     
#     def __init__(self, dataset_path, filtering_annotations_path=None, synthetic_annotation_path=None, 
#                  dataset_name="SyntheticDatasetPlaceHolder", cache_path="metadata_cache.pkl", 
#                  image_cache_dir=None):  # image_cache_dir is the directory to save verified images (optional)
#         data_type = "panoptic"
#         super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
        
#         # Load filtering annotations if provided
#         if self.filtering_annotations_path is not None:
#             filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
#             default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
#             default_annotation = {key: False for key in default_annotation.keys()}
#         else:
#             filtering_annotations = None
#             default_annotation = None

#         if synthetic_annotation_path is not None:
#             synthetic_annotation = json.load(open(synthetic_annotation_path, "r"))
#         else:
#             synthetic_annotation = {}

#         # Image cache directory for verified images (optional)
#         self.image_cache_dir = image_cache_dir

#         # Check if metadata cache exists:
#         if os.path.exists(cache_path):
#             print(f"Loading metadata from cache: {cache_path}")
#             with open(cache_path, "rb") as fp:
#                 self.metadata_table = pickle.load(fp)
#         else:
#             # If cache does not exist, create the metadata table
#             self.metadata_table = self._build_metadata_table(synthetic_annotation, filtering_annotations, default_annotation, dataset_name)
#             # Save to cache for future runs
#             with open(cache_path, "wb") as fp:
#                 pickle.dump(self.metadata_table, fp)
#             print(f"Metadata cache saved to: {cache_path}")

#         # Create a set of categories for later use
#         self.categories = set(self.metadata_table['category'])
    
#     def _process_folder(self, folder_name, synthetic_annotation, filtering_annotations, default_annotation, dataset_name):
#         """
#         Process one top-level directory and return a list of rows (dicts) for that folder.
#         Includes checking if each image is valid before adding it to the dataset.
#         """
#         rows = []
#         folder_path = os.path.join(self.dataset_path, folder_name)
#         if os.path.isdir(folder_path):
#             for subfolder_name in os.listdir(folder_path):
#                 subfolder_path = os.path.join(folder_path, subfolder_name)
#                 if os.path.isdir(subfolder_path):
#                     for file_name in os.listdir(subfolder_path):
#                         # Only process PNG files
#                         if file_name.endswith(".png"):
#                             image_path = os.path.join(self.dataset_path, folder_name, subfolder_name, file_name)
                            
#                             # Attempt to open and verify the image using Pillow
#                             try:
#                                 with Image.open(image_path) as img:
#                                     img.verify()  # Verify image integrity
#                             except Exception as e:
#                                 print(f"Skipping corrupted image: {image_path} ({e})")
#                                 continue  # Skip this image if it is corrupted

#                             # If image is verified and an image cache directory is set, save the verified image there
#                             if self.image_cache_dir:
#                                 # Reconstruct a similar folder structure in the cache directory
#                                 cache_folder = os.path.join(self.image_cache_dir, folder_name, subfolder_name)
#                                 os.makedirs(cache_folder, exist_ok=True)
#                                 cached_image_path = os.path.join(cache_folder, file_name)
#                                 # Copy the image file to the cache directory
#                                 copyfile(image_path, cached_image_path)
#                                 # Use the cached image path for further processing
#                                 image_path = cached_image_path

#                             image_subfolder_id = file_name.replace('.png', '').split("_")[-1]
#                             universial_idx = f"{dataset_name}_{folder_name}_{subfolder_name}_{image_subfolder_id}"
#                             query_idx = f"{folder_name}_{subfolder_name}"

#                             if query_idx in synthetic_annotation:
#                                 description = synthetic_annotation[query_idx]["description"]
#                                 short_phrase = synthetic_annotation[query_idx]["short_phrase"]
#                                 features = synthetic_annotation[query_idx]["features"]
#                                 # Get filtering annotation if exists
#                                 filtering_ann = (getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation)
#                                                    if filtering_annotations is not None else None)
#                                 rows.append({
#                                     'universal_idx': universial_idx,
#                                     'query_idx': query_idx,
#                                     'image_path': image_path,
#                                     'mask_path': image_path,  # Assuming mask_path is the same as image_path
#                                     'dataset_name': dataset_name,
#                                     'data_type': "panoptic",
#                                     'filtering_annotation': filtering_ann,
#                                     'category': folder_name,
#                                     'sub_category': subfolder_name,
#                                     'description': description,
#                                     'short_phrase': short_phrase,
#                                     'features': features
#                                 })
#         return rows

#     def _build_metadata_table(self, synthetic_annotation, filtering_annotations, default_annotation, dataset_name):
#         """
#         Build the metadata table from the dataset_path using multithreading.
#         """
#         all_rows = []
        
#         # Get the list of top-level directories
#         folders = [folder for folder in os.listdir(self.dataset_path) 
#                    if os.path.isdir(os.path.join(self.dataset_path, folder))]
        
#         # Define the number of threads you want to use
#         num_threads = 300  # Adjust as needed
        
#         # Use a ThreadPoolExecutor with a defined number of workers
#         with ThreadPoolExecutor(max_workers=num_threads) as executor:
#             # Submit jobs for each top-level folder
#             future_to_folder = {
#                 executor.submit(self._process_folder, folder, synthetic_annotation, filtering_annotations, default_annotation, dataset_name): folder
#                 for folder in folders
#             }
#             # Use tqdm to monitor progress
#             for future in tqdm(as_completed(future_to_folder), total=len(future_to_folder), desc="Processing folders"):
#                 folder = future_to_folder[future]
#                 try:
#                     rows = future.result()
#                     all_rows.extend(rows)
#                 except Exception as exc:
#                     print(f"Folder {folder} generated an exception: {exc}")
        
#         # Optionally, assign a sequential index across all rows
#         for idx, row in enumerate(all_rows):
#             row["idx_for_curr_dataset"] = idx
        
#         return pd.DataFrame(all_rows)









class SA1BDataset(BaseObjectDataset):
    def __init__(self, dataset_path, filtering_annotations_path):
        dataset_name = "sa1b_parsed" 
        data_type = "non-semantic"
        super().__init__(dataset_name, data_type, dataset_path, filtering_annotations_path)
        # get filtering annotation
        if self.filtering_annotations_path is not None:
            filtering_annotations = json.load(open(self.filtering_annotations_path, "r"))
            default_annotation = filtering_annotations[list(filtering_annotations.keys())[0]]
            default_annotation = {key: False for key in default_annotation.keys()}
        else:
            filtering_annotations = None
        
        
        # get metadata_table
        index_counter = 0
        rows = []
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if ".png" in file_name or ".jpg" in file_name:
                        image_path = os.path.join(self.dataset_path, folder_name, file_name)
                        universial_idx = f"{dataset_name}_{folder_name}_{file_name}"
                        if universial_idx in filtering_annotations:
                            rows.append({
                                'universal_idx': universial_idx,
                                'idx_for_curr_dataset': index_counter,
                                'category': folder_name,
                                'image_path': image_path,
                                'mask_path': image_path,  # Assuming mask_path is the same as image_path
                                'dataset_name': dataset_name,
                                'data_type': data_type,
                                'filtering_annotation': getting_filtering_annotation(universial_idx, filtering_annotations, default_annotation) if filtering_annotations is not None else None
                            })
                            index_counter += 1
        self.metadata_table = pd.DataFrame(rows)
        self.categories = set(self.metadata_table['category'])


class CustomizedDatasetOurFormat(BaseObjectDataset):
    def __init__(self, dataset_path, dataset_name, data_type):
        super().__init__(dataset_name, data_type, dataset_path)
        # get metadata_table
        index_counter = 0
        rows = []
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if ".png" in file_name:
                        image_path = os.path.join(self.dataset_path, folder_name, file_name)
                        rows.append({
                            'idx_for_curr_dataset': index_counter,
                            'category': folder_name,
                            'image_path': image_path,
                            'mask_path': image_path,  # Assuming mask_path is the same as image_path
                            'dataset_name': dataset_name,
                            'data_type': data_type
                        })
                        index_counter += 1
        self.metadata_table = pd.DataFrame(rows)
        self.categories = set(self.metadata_table['category'])

