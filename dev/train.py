"""
Usage:
    python train.py --config configs/standard_config.yaml
    python train.py --config configs/standard_config.yaml --resume experiments/standard_config/latest.pt
"""

import argparse
import yaml
import torch
from omegaconf import DictConfig

from data.dataset import FoodDataset
from data.dataloader import FoodDataLoader
from model.model import FoodClassifier
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FoodClassifier model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config file (e.g. configs/standard_config.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a checkpoint .pt file to resume training from"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and convert config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = DictConfig(config)

    print(f"Starting experiment: {config.experiment_name}")
    print(f"Config: {args.config}")

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Initialise dataset, dataloader, model, and trainer
    dataset = FoodDataset(config=config)
    dataloader = FoodDataLoader(dataset, config=config)
    model = FoodClassifier(config=config)
    model = torch.compile(model) # Use torch.compile for faster training if available
    trainer = Trainer(config=config, model=model, dataloader=dataloader)

    # Start training, optionally resuming from a checkpoint
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()