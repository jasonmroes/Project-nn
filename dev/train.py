"""
Usage:
    python train.py --config configs/standard_config.yaml
    python train.py --config configs/standard_config.yaml configs/k1_aug0_config.yaml
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
        nargs="+",
        required=True,
        help="Path(s) to the yaml config file (e.g. configs/standard_config.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a checkpoint .pt file to resume training from"
    )
    return parser.parse_args()

def train_one_config(config_path: str, resume_checkpoint: str = None):

    # Load and convert config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = DictConfig(config)

    print(f"\n{'='*60}")
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Initialise dataset, dataloader, model, and trainer
    dataset = FoodDataset(config=config)

    # Determine amount per class for weighted sampling
    # in train.py, after dataset is created
    label_counts = dataset.labels_df.iloc[:, 1].value_counts().sort_index()
    class_weights = (1.0 / label_counts).values
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)


    dataloader = FoodDataLoader(dataset, config=config)
    model = FoodClassifier(config=config)
    
    # Try compiling the model for efficiency on CUDA
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
        except Exception:
            pass

    trainer = Trainer(config=config, model=model, dataloader=dataloader, class_weights=class_weights)

    # Start training, optionally resuming from a checkpoint
    trainer.train(resume_from=resume_checkpoint)

def main():
    args = parse_args()

    for i, config_path in enumerate(args.config):
        resume_checkpoint = args.resume if i == 0 else None
        train_one_config(config_path, resume_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()