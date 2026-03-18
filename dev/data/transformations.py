import torchvision.transforms as transforms
import torch

def rotate_translate_flip(image: torch.Tensor, p:float = 0.5, degrees: int = 20, translate: tuple = (0.1, 0.1)) -> torch.Tensor:
    """Apply random rotation up to 'degrees', translation up to (a * image_width, b * image_height), 
    and flips horizontally and vertically with probability 'p'."""
    tranforms = transforms.Compose([
        transforms.RandomAffine(degrees, translate), # Rotate up to 20 degrees left or right, translate up to 0.1 * img_dimension
        transforms.RandomHorizontalFlip(p), # 50/50 chance of horizontal flip
        transforms.RandomVerticalFlip(p), # 50/50 chance of vertical flip
    ])
    return tranforms(image)