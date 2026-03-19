import os
import pandas as pd
import torch
from torchvision import transforms
from omegaconf import DictConfig # loading config yaml
from PIL import Image # image processing and transformation
import random
import data.transformations # custom transformations

# Dataset class to load the information stored in /data and make it
# easily accessible to the rest of the project.

# TODO: implement transformations to 'expand' dataset 

class FoodDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading food images and their labels
    args:
        config: DictConfig = None, Use this to load config yaml
        data_dir: path to dataset, set in yaml
        labels_path: path to csv file with the image filenameslabels, set in yaml
        image_dir: path to images, set in yaml
        indices: optional list of indices to load specific samples from the dataset (e.g., for train/val split)
        correspond to the number of the image and label-1 (0 -> train_1.png)
        val_fraction: fraction of the dataset to be used for validation (if indices are not provided)
        augment_transform: transformations to be applied to a fraction of images (yaml) beyond resizing and normalization.
    """

    def __init__(
            self,
            config: DictConfig = None, # Use this to load config yaml
            data_dir: str = "data/",
            labels_path: str = "data/train_labels.csv",
            image_dir: str = "data/train_set/train_set/train_set/",
            image_shape: list = [128, 128],
            indices: list =  None, # Optional: load samples indicated by the indices from train/val, as split by split.py
            val_fraction: float = 0.1,
            augment_transform: transforms.Compose = None, # TODO reference yaml
            augment_fraction: float = -1.0 # no data augmentation unless specifically asked
            ):

        if config:
            data_dir = config.data.data_path
            labels_path = config.data.labels_path
            image_dir = config.data.image_dir
            val_fraction = config.data.val_fraction
            image_shape = config.data.image_shape
            augment_fraction = config.data.augment_fraction
            augment_transform = getattr(data.transformations, config.data.augmentation_function) # Get the augmentation function from the transformations module based on the name in the yaml

        self.data_dir = data_dir
        self.labels_path = labels_path
        self.image_dir = image_dir
        self.image_shape = image_shape
        self.val_fraction = val_fraction
        self.indices = indices
        self.augment_transform = augment_transform
        self.augment_fraction = augment_fraction
        self.labels_df = pd.read_csv(labels_path) # Loads **all** labels from labels.csv

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.labels_df)

    def __getitem__(self, idx):

        if self.indices is not None:
            idx = self.indices[idx] # Map the provided index to the actual index in the labels dataframe

        image_name = self.labels_df.iloc[idx, 0] # the first column contains image filenames
        label = self.labels_df.iloc[idx, 1] - 1 # the second column contains labels, convert 1-indexed to 0-indexed


        # Data directory + correct folder + specific image
        image_path = os.path.join(self.image_dir, str(image_name))

        # Load the Image and convert to RGB if not already
        image = Image.open(image_path).convert("RGB")

        # apply augment_transform if provided to a fraction of the images in the training set (as specified in the yaml)
        if self.augment_transform:
            if random.random() < self.augment_fraction:
                image = self.augment_transform(image)
        
        image = data.transformations.standardise(self.image_shape, image)


        return image, label


# Test code
# if __name__ == "__main__":
#     dataset = FoodDataset()
#     print(f"Dataset length: {len(dataset)}")
#     image, label = dataset[0]
#     print(f"First image shape: {image.size}, Label: {label}")
#     image.show()