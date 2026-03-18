from data.dataloader import FoodDataLoader
from data.dataset import FoodDataset
from model.model import FoodClassifier

import torch
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig

# Trainer code generated with Claude
class Trainer:
    def __init__(self, model: nn.Module, dataloader: FoodDataLoader, config: DictConfig = None):
        if config:
            self.model = model
            self.dataloader = dataloader
            self.criterion = nn.CrossEntropyLoss() # simple choice for multi-class classification
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
            self.num_epochs = config.training.epochs
        else:
            raise ValueError("Config must be provided for Trainer initialization.")

    def train_single_epoch(self):
        """Train the model for one epoch on the training data."""
        self.model.train()  # Set model to training mode
        total_loss = 0.0

        for fold, train_loader, val_loader in self.dataloader.get_k_fold_dataloaders():  # Get the loaders from the first fold
            self.optimizer.zero_grad()  # Clear gradients
            seen = 0 # For tracking progress within fold

            for images, labels in train_loader:
                outputs = self.model(images)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                total_loss += loss.item() * images.size(0)  # Accumulate loss
                seen += images.shape[0] # For tracking progress within fold
                print(f"Fold progress: {seen}/{len(train_loader.dataset)}", end="\r")

            print(f"Fold {fold}: Training loss = {total_loss / len(train_loader.dataset):.4f}")

    def evaluate(self, val_loader):
        """Evaluate the model on the validation data."""
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # No need to compute gradients during evaluation
            for images, labels in val_loader:
                outputs = self.model(images)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # Count correct predictions

        accuracy = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    def train(self):
        """Train the model for the specified number of epochs, evaluating on validation data after each epoch."""
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.train_single_epoch()  # Train for one epoch

            # Evaluate on validation data from the first fold (for simplicity)
            _, train_loader, val_loader = next(self.dataloader.get_k_fold_dataloaders())
            self.evaluate(val_loader)

if __name__ == "__main__":
    # Quick sanity check for the Trainer class
    with open("configs/standard_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config = DictConfig(config) # Convert to DictConfig for consistency
    data = FoodDataset(config=config) # Use the config to initialize the dataset
    num_classes = len(data.labels_df['label'].unique()) # Dynamically determine number of classes from the dataset labels
    model = FoodClassifier(num_classes=num_classes, config=config)
    dataloader = FoodDataLoader(data, batch_size=4, shuffle=True)

    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.train_single_epoch()  # Just test one epoch for sanity check
