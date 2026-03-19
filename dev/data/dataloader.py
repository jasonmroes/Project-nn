import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig # for loading config yaml
from data.dataset import FoodDataset
from data.split import split_train_val, split_kfold


# DataLoader class to create batches of data for training and validation.

class FoodDataLoader:
    """
    DataLoader class for creating batches of food images and their labels
    args:
        dataset: FoodDataset, the dataset to load from
        config: DictConfig = None, Use this to load config yaml
        indices: list = None, Optional: load samples indicated by the indices from train/val, as split by split.py
        batch_size: int = 32, number of samples per batch
        shuffle: bool = True, whether to shuffle the data at every epoch
        num_workers: int = 0, number of subprocesses to use for data loading
        k: int = 4, Number of folds for k-fold cross-validation (if using k-fold)
    """
    def __init__(
            self,
            dataset: FoodDataset,
            config: DictConfig = None,
            indices: list = None,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            k: int = 4
            ):
        if config:
            self.config = config
            batch_size = config.data.batch_size
            shuffle = config.data.shuffle
            num_workers = config.data.num_workers
            k = config.data.k

        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.k = k

    # def get_dataloader(
    #         self,
    #         data_dir:str = "data/",
    #         labels_path:str = "data/train_labels.csv",
    #         image_dir:str = "data/train_set/train_set/train_set/"
    #         ):
            
    #     return DataLoader.DataLoader(
    #         self.dataset,
    #         batch_size=self.batch_size,
    #         shuffle=self.shuffle,

    #         num_workers=self.num_workers
    #     )
    
    def _make_loader(
            self,
            indices: list,
            shuffle: bool,
            is_validation: bool = False
            ) -> DataLoader:
        """
        Helper function to create a DataLoader for a given set of indices and shuffle setting
            args:
            indices: list of indices to load from the dataset
            shuffle: whether to shuffle the data in this loader
            data_dir: path to dataset, set in yaml
            labels_path: path to csv file with the image filenameslabels, set in yaml
            image_dir: path to images, set in yaml
            """
        dataset = FoodDataset(
            config=self.config,
            indices=indices,
            augment_fraction=0.0 if is_validation else self.config.data.augment_fraction, # Only augment training data
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_train_val_dataloaders(
            self,
            data_dir:str = "data/",
            labels_path:str = "data/train_labels.csv",
            image_dir:str = "data/train_set/train_set/train_set/",
            val_fraction: float = 0.1,
            seed: int = 42
            ) -> tuple[DataLoader, DataLoader]:
            """
            Return (train_loader, val_loader) using a k-fold split as determined in split.py
                args:
                data_dir: path to dataset, set in yaml
                labels_path: path to csv file with the image labels, set in yaml
                image_dir: path to images, set in yaml
                val_fraction: fraction of the dataset to be used for validation (if indices are not provided"""

            train_indices, val_indices = split_train_val(
                labels_path=labels_path,
                val_fraction=val_fraction,
                seed=seed,
            )
            train_loader = self._make_loader(train_indices, self.shuffle, False)
            val_loader   = self._make_loader(val_indices,   False, True)
            return train_loader, val_loader

    #### K-fold cross-validation ####

    def get_k_fold_dataloaders(
            self,
            data_dir: str    = "data/",
            labels_path: str = "data/train_labels.csv",
            image_dir: str   = "data/train_set/train_set/train_set/",
            seed: int = 42
            ):
        """Yield (fold, train_loader, val_loader) for each of the k folds.
 
        Usage:
            for fold, train_loader, val_loader in food_dataloader.get_k_fold_dataloaders():
                # train and evaluate for this fold ...
        """
        splits = split_kfold(
            labels_path=labels_path,
            k=self.k,
            shuffle=self.shuffle,
            seed=seed,
        )
        for fold, (train_indices, val_indices) in enumerate(splits):
            train_loader = self._make_loader(train_indices, self.shuffle, data_dir, labels_path, image_dir, is_validation=False)
            val_loader   = self._make_loader(val_indices,   False,        data_dir, labels_path, image_dir, is_validation=True)
            yield fold, train_loader, val_loader


# Test the dataloader
if __name__ == "__main__":
    dataset = FoodDataset()
    dataloader = FoodDataLoader(dataset, batch_size=4, shuffle=False)

    # Test getting 1 train and 1 val dataloader
    train_loader, val_loader = dataloader.get_train_val_dataloaders(train_indices=[0,1,2,3], val_indices=[4,5,6,7])
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}, Batch of labels shape: {labels.shape}")
        break # Only test the first batch

    # Test k-fold dataloader
    for fold, train_loader, val_loader in dataloader.get_k_fold_dataloaders():
        print(f"Fold {fold + 1}")
        for images, labels in train_loader:
            print(f"  train batch — images: {images.shape}, labels: {labels.shape}")
            break
        for images, labels in val_loader:
            print(f"  val   batch — images: {images.shape}, labels: {labels.shape}")
            break