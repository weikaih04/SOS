import pandas as pd

class DataManager():
    def __init__(self, available_object_datasets, available_background_datasets, filtering_setting):
        # Process object datasets.
        self.available_object_datasets = available_object_datasets
        self.filtered_object_data = {}  # To store filtered data for each object dataset.
        self.accumulated_object_metadata_table = pd.DataFrame()
        unified_object_categories = set()

        for dataset_name, curr_dataset in self.available_object_datasets.items():
            # Retrieve the original metadata table.
            metadata_table = curr_dataset.return_metadata_table()
            count_before = len(metadata_table)
            
            # Apply filtering based on the provided settings.
            if metadata_table['filtering_annotation'].isna().any():
                filtered_table = metadata_table
                count_after = count_before
            else:
                filter_mask = pd.Series(True, index=metadata_table.index)
                for metric, condition in filtering_setting.items():
                    if condition == "filter":
                        filter_mask &= metadata_table['filtering_annotation'].apply(
                            lambda x: x.get(metric, False) if pd.notna(x) else False
                        )
                filtered_table = metadata_table[filter_mask]
                count_after = len(filtered_table)

            # Print counts for this dataset.
            print(f"Object dataset '{dataset_name}':")
            print(f"  Count before filtering: {count_before}")
            print(f"  Count after filtering: {count_after}")

            # Store the filtered table for later access and accumulate in one table.
            self.filtered_object_data[dataset_name] = filtered_table
            self.accumulated_object_metadata_table = pd.concat(
                [self.accumulated_object_metadata_table, filtered_table]
            )

            # Collect unified object categories.
            unified_object_categories.update(curr_dataset.categories)

        # Prepare a sorted list of unified object categories and a mapping to indices.
        self.unified_object_categories = sorted(list(unified_object_categories))
        self.unified_object_categories_to_idx = {category: idx for idx, category in enumerate(self.unified_object_categories)}

        # Process background datasets.
        self.available_background_datasets = available_background_datasets
        self.accumulated_background_metadata_table = pd.DataFrame()
        self.background_data = {}  # Store each background dataset's metadata table.
        for dataset_name, curr_dataset in self.available_background_datasets.items():
            metadata_table = curr_dataset.return_metadata_table()
            self.background_data[dataset_name] = metadata_table
            self.accumulated_background_metadata_table = pd.concat(
                [self.accumulated_background_metadata_table, metadata_table]
            )
            print(f"Background dataset '{dataset_name}' has {len(metadata_table)} records.")

        print("Finished processing object and background datasets.")
    
    # object
    def query_object_metadata(self, query):
        return self.accumulated_object_metadata_table.query(query)
    
    def get_object_by_metadata(self, metadata):
        dataset_name = metadata["dataset_name"]
        idx_for_curr_dataset = metadata["idx_for_curr_dataset"]
        return self.available_object_datasets[dataset_name][idx_for_curr_dataset]
    
    def get_random_object_metadata(self, rng):
        # First uniformly select an object dataset.
        dataset_name = rng.choice(list(self.filtered_object_data.keys()))
        dataset_table = self.filtered_object_data[dataset_name]
        # Then uniformly sample from that dataset's metadata table.
        idx = rng.choice(len(dataset_table))
        selected_metadata = dataset_table.iloc[idx].copy()
        # Add dataset information to the metadata.
        selected_metadata["dataset_name"] = dataset_name
        return selected_metadata
    
    # background
    def query_background_metadata(self, query):
        return self.accumulated_background_metadata_table.query(query)
    
    def get_background_by_metadata(self, metadata):
        dataset_name = metadata["dataset_name"]
        idx_for_curr_dataset = metadata["idx_for_curr_dataset"]
        return self.available_background_datasets[dataset_name][idx_for_curr_dataset]
    
    def get_random_background_metadata(self, rng):
        # First uniformly select a background dataset.
        dataset_name = rng.choice(list(self.background_data.keys()))
        metadata_table = self.background_data[dataset_name]
        # Then uniformly sample from that dataset's metadata table.
        idx = rng.choice(len(metadata_table))
        selected_metadata = metadata_table.iloc[idx].copy()
        # Add dataset information to the metadata.
        selected_metadata["dataset_name"] = dataset_name
        return selected_metadata

    def get_random_object_metadata_by_category(self, rng, category):
        df = self.accumulated_object_metadata_table
        # vectorized filter
        cat_df = df[df['category'] == category]
        if cat_df.empty:
            raise ValueError(f"No objects found for category '{category}'")
        # sample a single row by integer location
        i = rng.integers(0, len(cat_df))
        return cat_df.iloc[i].copy()