from dev.data.dataset import FoodDataset
from dev.data.dataloader import FoodDataLoader
from PIL import Image
import torch

##### Dataset tests #####

def test_retrieve_first_and_last_items():
    """Test whether the dataset __getitem__ method correctly retrieves an image and its label"""
    dataset = FoodDataset()

    image, label = dataset[0] # Retrieve the first item in the dataset
    assert isinstance(image, torch.Tensor), "The retrieved image should be a torch tensor"
    assert label == 21, "First image label in the train_set is 21"

    image, label = dataset[-1]
    assert isinstance(image, torch.Tensor), "The retrieved image should be a torch tensor"
    assert label == 24, "Last image label in the train_set is 24"

def test_retrieve_first_three_items():
    """Test whether the dataset __getitem__ method correctly retrieves the first three images and their labels"""
    dataset = FoodDataset()

    for i in range(3):
        image, label = dataset[i]
        assert isinstance(image, torch.Tensor), f"Image at index {i} is not a tensor"

##### Dataloader tests ######

def test_dataloader():
    """Test whether the dataloader correctly retrieves a batch of images and labels"""
    dataset = FoodDataset()
    dataloader = FoodDataLoader(dataset, batch_size=4, shuffle=False)

    # first 4 images are train set, next 4 the val_set
    train_loader, val_loader = dataloader.get_train_val_dataloaders(train_indices=[0,1,2,3], val_indices=[4,5,6,7])

    # Test the training dataloader
    for images, labels in train_loader:
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[0] == 4, "Batch size should be 4"
        assert labels.shape[0] == 4, "Batch size should be 4"
        break # Only test the first batch
    
    # Test the training validation dataloader
    for images, labels in val_loader:
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[0] == 4, "Batch size should be 4"
        assert labels.shape[0] == 4, "Batch size should be 4"
        break # Only test the first batch