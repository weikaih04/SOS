from .base_datasets import BaseBackgroundDataset
import os
import glob
import pandas as pd


class BG20KDataset(BaseBackgroundDataset):
    def __init__(self, dataset_path):
        dataset_name = "BG20k" 
        data_type = "background"
        super().__init__(dataset_name, data_type, dataset_path)
        # get metadata_table
        rows = []
        self.metadata_table = pd.DataFrame(columns=['idx_for_curr_dataset', 'category', 'image_path', 'dataset_name', 'data_type'])
        index_counter = 0
        for image_path in glob.glob(self.dataset_path + "/train/*.jpg"):
            rows.append({
                'idx_for_curr_dataset': index_counter,
                'category': "no category",
                'image_path': image_path,
                'dataset_name': dataset_name,
                'data_type': data_type
            })
            index_counter += 1
        self.metadata_table = pd.DataFrame(rows)

class NoBGDataset(BaseBackgroundDataset):
    pass