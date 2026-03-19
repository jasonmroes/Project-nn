"""
inference.py — run inference on a folder of test images and save predictions to a CSV.

Usage:
    python inference.py --checkpoint experiments/standard_config/best.pt
                        --image_dir data/test_set/
                        --config configs/standard_config.yaml
                        --output predictions.csv
"""

import argparse
import os
import yaml
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from omegaconf import DictConfig

from model.model import FoodClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained FoodClassifier checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint .pt file (e.g. experiments/standard_config/best.pt)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the folder containing test images"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config file used during training — needed for image size and num_classes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to the output CSV file (default: predictions.csv)"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: DictConfig, device: torch.device) -> FoodClassifier:
    """Load the FoodClassifier from a checkpoint file."""
    model = FoodClassifier(config=config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from '{checkpoint_path}' (epoch {checkpoint.get('epoch', '?')})")
    return model


def standardise(image_shape: list) -> transforms.Compose:
    """the same standardisation transform used during training. No augmentation"""
    return transforms.Compose([
        transforms.Resize(tuple(image_shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_image_paths(image_dir: str) -> list[str]:
    """Return a sorted list of all image file paths in the given directory."""
    paths = [
        entry.name for entry in os.scandir(image_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in '{image_dir}'. "
                                f"Supported extensions: {IMAGE_EXTENSIONS}")
    return sorted(paths)


def run_inference(
        model: FoodClassifier,
        image_dir: str,
        image_names: list[str],
        transform: transforms.Compose,
        device: torch.device,
        ) -> list[int]:
    """Run the model on each image and return a list of predicted class labels.

    Labels are converted back to 1-indexed to match the original dataset convention
    (training subtracted 1 to make them 0-indexed for CrossEntropyLoss).
    """
    predictions = []

    with torch.no_grad():
        for i, img_name in enumerate(image_names):
            image_path = os.path.join(image_dir, img_name)

            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension -> (1, C, H, W)

            logits = model(tensor)                             # (1, num_classes)
            predicted_class = torch.argmax(logits, dim=1).item() + 1  # +1 to restore 1-indexing

            predictions.append(predicted_class)

            # Progress indicator
            print(f"Processed {i + 1}/{len(image_names)}: {img_name} -> class {predicted_class}", end="\r")

    print()  # newline after progress line
    return predictions


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = DictConfig(config)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Build the same transform pipeline used during training
    transform = standardise(config.data.image_shape)

    # Collect test image filenames
    image_names = get_image_paths(args.image_dir)
    print(f"Found {len(image_names)} images in '{args.image_dir}'")

    # Run inference
    predictions = run_inference(model, args.image_dir, image_names, transform, device)

    # Save to CSV
    results_df = pd.DataFrame({
        "img_name": image_names,
        "label":    predictions,
    })
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to '{args.output}'")


if __name__ == "__main__":
    main()