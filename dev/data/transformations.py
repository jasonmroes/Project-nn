import torchvision.transforms as transforms
import torch
from PIL.Image import Image

# Transform images to be the same size and normalised
def standardise(image_shape: list, image: Image) -> torch.Tensor:
    standardise = transforms.Compose([
        transforms.Resize(tuple(image_shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return standardise(image)

def rotate_translate_flip(image: Image, p:float = 0.5, degrees: int = 20, translate: tuple = (0.1, 0.1)) -> Image:
    """Apply random rotation up to 'degrees', translation up to (a * image_width, b * image_height), 
    and flips horizontally and vertically with probability 'p'."""
    tranforms = transforms.Compose([
        transforms.RandomAffine(degrees, translate), # Rotate up to 20 degrees left or right, translate up to 0.1 * img_dimension
        transforms.RandomHorizontalFlip(p), # 50/50 chance of horizontal flip
    ])
    return tranforms(image)