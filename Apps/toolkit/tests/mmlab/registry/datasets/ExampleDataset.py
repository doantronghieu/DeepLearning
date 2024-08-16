import add_packages
import torch
import numpy as np
from typing import List, Dict
from mmengine.dataset import BaseDataset 
from mmengine.structures import BaseDataElement
from mmengine.registry import DATASETS
from toolkit.mmlab.engine.bases.data import DataManager

@DATASETS.register_module()
class LinearRegressionDataset(BaseDataset):
    def __init__(self, num_samples=1000, num_features=1, noise=0.1):
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise = noise
        
        # Generate synthetic data
        self.X = np.random.randn(num_samples, num_features)
        true_coefficients = np.random.randn(num_features)
        self.y = np.dot(self.X, true_coefficients) + np.random.randn(num_samples) * noise
        
        # Call super().__init__() after generating the data
        super().__init__()

    def load_data_list(self):
        """Override this method to return a list of data samples."""
        return [
            {
                'features': self.X[i],
                'target': self.y[i]
            }
            for i in range(self.num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return BaseDataElement(
            features=torch.FloatTensor(self.X[idx]),
            target=torch.FloatTensor([self.y[idx]])
        )

class LinearRegressionDataManager(DataManager):
    def collate_fn(self, batch: List[BaseDataElement]) -> Dict[str, torch.Tensor]:
        """Collate function for linear regression data."""
        features = []
        targets = []
        
        for data_element in batch:
            features.append(data_element.features)
            targets.append(data_element.target)
        
        # Stack features and targets into tensors
        feature_tensor = torch.stack(features)
        target_tensor = torch.cat(targets)
        
        return {
            'features': feature_tensor,
            'targets': target_tensor
        }

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a batch of data for linear regression."""
        # In this simple example, we just return the batch as-is
        return batch

# Configuration for the dataset
dataset_config = {
    'type': 'LinearRegressionDataset',
    'num_samples': 1000,
    'num_features': 3,
    'noise': 0.1
}

# Configuration for the dataloader
dataloader_config = {
    'batch_size': 32,
    'num_workers': 0  # Set to 0 for debugging
}

# Create the data manager
linear_regression_dm = LinearRegressionDataManager(
    dataset_configs=dataset_config,
    dataloader_config=dataloader_config
)

# Prepare the data
linear_regression_dm.prepare_data()

# Now you can use the data manager to get batches
for batch in linear_regression_dm:
    features: torch.Tensor = batch['features']
    targets: torch.Tensor = batch['targets']
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")

# You can also get a single batch like this:
single_batch = linear_regression_dm.get_batch()
print(f"Single batch features shape: {single_batch['features'].shape}")

# TODO: Write Pytest