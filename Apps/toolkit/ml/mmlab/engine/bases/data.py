from typing import Any, Dict, List, Union, Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from mmengine.dataset import BaseDataset, DefaultSampler
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose
from mmengine.dataset import ConcatDataset, RepeatDataset, ClassBalancedDataset
from mmengine.structures import BaseDataElement

class DataManager(ABC):
    def __init__(
        self,
        dataset_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        dataloader_config: Dict[str, Any],
        transform_configs: Optional[List[Dict[str, Any]]] = None,
        dataset_wrapper: Optional[str] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs]
        self.dataloader_config = dataloader_config
        self.transform_configs = transform_configs
        self.dataset_wrapper = dataset_wrapper
        self.sampler_config = sampler_config or {'type': 'DefaultSampler', 'shuffle': True}
        
        self.datasets: List[BaseDataset] = []
        self.dataloader: Optional[DataLoader] = None
        self.transforms: Optional[Compose] = None

    def build_transforms(self) -> Compose:
        """Build transforms based on the configuration."""
        if not self.transform_configs:
            return Compose([])
        
        built_transforms = []
        for transform_config in self.transform_configs:
            transform_type = transform_config.pop('type')
            transform = TRANSFORMS.build(dict(type=transform_type, **transform_config))
            built_transforms.append(transform)
        return Compose(built_transforms)

    def build_datasets(self) -> List[BaseDataset]:
        """Build datasets based on the configurations."""
        datasets = []
        for config in self.dataset_configs:
            dataset_type = config.pop('type')
            dataset = DATASETS.build(dict(type=dataset_type, **config))
            datasets.append(dataset)
        return datasets

    def apply_dataset_wrapper(self, datasets: List[BaseDataset]) -> BaseDataset:
        """Apply dataset wrapper if specified."""
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        if self.dataset_wrapper:
            if self.dataset_wrapper == 'RepeatDataset':
                dataset = RepeatDataset(dataset, times=self.dataloader_config.get('times', 1))
            elif self.dataset_wrapper == 'ClassBalancedDataset':
                dataset = ClassBalancedDataset(dataset, oversample_thr=self.dataloader_config.get('oversample_thr', 1e-3))
        
        return dataset

    def build_sampler(self, dataset: BaseDataset) -> Any:
        """Build sampler based on the configuration."""
        sampler_type = self.sampler_config.pop('type')
        if sampler_type == 'DefaultSampler':
            return DefaultSampler(dataset, **self.sampler_config)
        else:
            return DATASETS.build(dict(type=sampler_type, dataset=dataset, **self.sampler_config))

    def prepare_data(self) -> None:
        """Prepare the dataset, transforms, and dataloader."""
        self.transforms = self.build_transforms()
        self.datasets = self.build_datasets()
        dataset = self.apply_dataset_wrapper(self.datasets)
        
        sampler = self.build_sampler(dataset)
        
        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.dataloader_config.get('batch_size', 1),
            num_workers=self.dataloader_config.get('num_workers', 0),
            collate_fn=self.collate_fn,
            **{k: v for k, v in self.dataloader_config.items() if k not in ['batch_size', 'num_workers']}
        )

    def get_data_element(self, idx: int) -> BaseDataElement:
        """Get a single data element from the dataset."""
        if not self.datasets:
            raise ValueError("Datasets have not been prepared. Call prepare_data() first.")
        return self.datasets[0][idx]  # Assumes at least one dataset

    def apply_transforms(self, data_element: BaseDataElement) -> BaseDataElement:
        """Apply transforms to a single data element."""
        if self.transforms is None:
            raise ValueError("Transforms have not been prepared. Call prepare_data() first.")
        return self.transforms(data_element)

    def get_batch(self) -> List[BaseDataElement]:
        """Get a batch of data elements from the dataloader."""
        if self.dataloader is None:
            raise ValueError("Dataloader has not been prepared. Call prepare_data() first.")
        return next(iter(self.dataloader))

    @abstractmethod
    def collate_fn(self, batch: List[BaseDataElement]) -> Any:
        """Collate function for the dataloader. This method should be implemented
        by subclasses to handle domain-specific collation."""
        pass

    @abstractmethod
    def process_batch(self, batch: Any) -> Any:
        """Process a batch of data elements. This method should be implemented
        by subclasses to handle domain-specific processing."""
        pass

    def __iter__(self):
        if self.dataloader is None:
            raise ValueError("Dataloader has not been prepared. Call prepare_data() first.")
        return iter(self.dataloader)

    def __len__(self):
        if self.dataloader is None:
            raise ValueError("Dataloader has not been prepared. Call prepare_data() first.")
        return len(self.dataloader)

