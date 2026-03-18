from data.dataset import FoodDataset
from data.dataloader import FoodDataLoader
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig 

# Model code generated with Claude

### Convolutional Neural Network ###

class ConvBlock(nn.Module):
    """A single conv -> BN -> ReLU -> (optional) MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FoodClassifier(nn.Module):
    """
    A from-scratch CNN for food image classification.

    Architecture:
        Input:  (B, 3, 224, 224)
        Block 1: 3   -> 32  channels, pool  -> (B, 32,  112, 112)
        Block 2: 32  -> 64  channels, pool  -> (B, 64,   56,  56)
        Block 3: 64  -> 128 channels, pool  -> (B, 128,  28,  28)
        Block 4: 128 -> 256 channels, pool  -> (B, 256,  14,  14)
        Block 5: 256 -> 256 channels, pool  -> (B, 256,   7,   7)
        GAP:                                -> (B, 256,   1,   1)
        Flatten:                            -> (B, 256)
        FC + Dropout:                       -> (B, 128)
        Output:                             -> (B, num_classes)

    Args:
        num_classes: Number of food categories to predict.
        dropout: Dropout probability before the final FC layer.
    """

    def __init__(self, config: DictConfig, num_classes: int, dropout: float = 0.5):
        super().__init__()

        if config:
            num_classes = config.data.classes
            dropout = config.model.dropout_rate

        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # -> (B, 32,  112, 112)
            ConvBlock(32,  64,  pool=True),   # -> (B, 64,   56,  56)
            ConvBlock(64,  128, pool=True),   # -> (B, 128,  28,  28)
            ConvBlock(128, 256, pool=True),   # -> (B, 256,  14,  14)
            ConvBlock(256, 256, pool=True),   # -> (B, 256,   7,   7)
        )

        # Global Average Pooling collapses spatial dims to 1x1,
        # making the model robust to slight input size variations
        # and reducing parameters vs a raw Flatten.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # spatial feature maps
        x = self.gap(x)        # (B, 256, 1, 1)
        x = self.classifier(x) # (B, num_classes)
        return x               # raw logits — loss fn applies softmax


if __name__ == "__main__":
    # Quick sanity check — mirrors your test convention
    with open("configs/standard_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config = DictConfig(config) # Convert to DictConfig for consistency
    data = FoodDataset(config=config) # Use the config to initialize the dataset
    num_classes = len(data.labels_df['label'].unique()) # Dynamically determine number of classes from the dataset labels
    model = FoodClassifier(num_classes=num_classes)

    dummy = torch.randn(4, 3, 224, 224)  # batch of 4 images
    logits = model(dummy)

    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {logits.shape}")   # expect (4, 50)
    assert logits.shape == (4, num_classes), "Output shape mismatch!"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")