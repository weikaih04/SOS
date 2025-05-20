from PIL import Image
import numpy as np
from torch.utils.data import Dataset


def convert_rgba_to_rgb_and_mask(rgba_image: Image):
    # Open the RGBA image
    if rgba_image.mode != "RGBA":
        raise ValueError("Input image must be in RGBA mode")    
    # Convert to RGB
    rgb_image = rgba_image.convert("RGB")
    
    # Create a mask based on the alpha channel
    alpha_channel = np.array(rgba_image.getchannel('A'))
    mask = Image.fromarray(alpha_channel)  # This will be a grayscale image
    return rgb_image, mask


def get_bounding_box(mask):
    pass

def get_image_and_mask(image_path, mask_path):
    if image_path == mask_path:
        image_with_mask = Image.open(image_path)
        if image_with_mask.mode == "RGBA":
            return convert_rgba_to_rgb_and_mask(image_with_mask)
        else:
            raise ValueError("Image and mask are the same, but image is not in RGBA mode")
    else:
        return Image.open(image_path), Image.open(mask_path)    


class BaseObjectDataset(Dataset):
    def __init__(self, dataset_name, data_type, dataset_path, filtering_annotations_path=None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_type = data_type
        self.metadata_table = None
        self.filtering_annotations_path = filtering_annotations_path

    
    def __len__(self):
        return len(self.metadata_table)

    def __getitem__(self, index):
        data_package = {}
        metadata = self.metadata_table[self.metadata_table['idx_for_curr_dataset'] == index].iloc[0]
        data_package['metadata'] = metadata
        data_package['image'], data_package['mask'] = get_image_and_mask(metadata['image_path'], metadata['mask_path'])
        return data_package

    def return_metadata_table(self):
        # metadata contains the information of each mask, like its cateogory, path.
        # each mask store as a row in the metadata table
        return self.metadata_table
    

class BaseBackgroundDataset(Dataset):
    def __init__(self, dataset_name, data_type, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_type = data_type
        self.metadata_table = None
    
    def __len__(self):
        return len(self.metadata_table)

    def __getitem__(self, index):
        data_package = {}
        metadata = self.metadata_table[self.metadata_table['idx_for_curr_dataset'] == index].iloc[0]
        data_package['metadata'] = metadata
        data_package['image'] = Image.open(metadata['image_path']).convert("RGB") 
        return data_package

    def return_metadata_table(self):
        # metadata contains the information of each mask, like its cateogory, path.
        # each mask store as a row in the metadata table
        return self.metadata_table