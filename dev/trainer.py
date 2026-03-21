from data.dataloader import FoodDataLoader
from data.dataset import FoodDataset
from model.model import FoodClassifier

import yaml
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from datetime import datetime

# Improve performance on gpu by lowering precision (negligible for CNN)
torch.set_float32_matmul_precision('high')

# Trainer code generated with Claude
class Trainer:
    def __init__(self, config: DictConfig, model: nn.Module, dataloader: FoodDataLoader, class_weights: torch.Tensor):
        """Trainer class to handle the training loop, evaluation, checkpointing, and logging.
        Args:
            model: The neural network model to be trained.
            dataloader: The DataLoader providing training and validation data.
            config: Configuration object containing hyperparameters and settings.
        """

        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        self.num_epochs = config.training.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device)) # simple choice for multi-class classification


        # Step on plateau lr scheduler to reduce learning rate if validation accuracy plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',       # we're monitoring val accuracy (higher is better)
            factor=config.training.scheduler.gamma,         # reduce the lr when plateauing by factor of ...
            patience=config.training.scheduler.step_size,   # wait ... epochs without improvement before reducing
        )
        
        # Checkpointing — save to experiments/checkpoints/ by default
        self.checkpoint_dir = config.training.get("save_dir", "experiments/checkpoints/")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_accuracy = 0.0
 

        # TensorBoard writer
        run_name = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(config.training.get("log_dir", "runs/"), run_name)
        os.makedirs(log_dir, exist_ok=True)
        print("Logging TensorBoard data to:", log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
 
        # Global step counter so TensorBoard x-axis is always increasing
        # across folds and epochs
        self.global_step = 0
    
    def save_checkpoint(self, epoch: int, val_accuracy: float):
        """Save model weights and training state to disk.

        Two files are maintained:
          - latest.pt  : overwritten every epoch (safe resume point)
          - best.pt    : overwritten only when val_accuracy improves
        """
        state = {
            "epoch":          epoch,
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_accuracy":   val_accuracy,
        }
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(state, latest_path)
 
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)
            print(f"New best checkpoint saved (val_accuracy={val_accuracy:.4f})")
    
    def  load_checkpoint(self, path: str):
        """Resume training from a checkpoint file."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_val_accuracy = checkpoint.get("val_accuracy", 0.0)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint '{path}' (epoch {checkpoint['epoch']})")
        return start_epoch



    def _train_single_epoch(self,train_loader: FoodDataLoader, epoch: int, fold: int):
        """Train the model for one epoch on the training loader."""
        self.model.train()  # Set model to training mode
        fold_loss = 0.0

        seen = 0 # For tracking progress within fold

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)  # Move data to the same device as the model
            self.optimizer.zero_grad()  # Clear gradients
            outputs = self.model(images)  # Forward pass
            loss = self.criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights


            fold_loss += loss.item() * images.size(0)  # Accumulate loss
            seen += images.shape[0] # For tracking progress within fold
            self.global_step += 1 # increment global step for TensorBoard

            
            # Per-batch loss — useful for spotting instability early
            self.writer.add_scalar(
                f"Loss/train_batch (fold {fold})",
                loss.item(),
                self.global_step
            )
            print(f"Fold {fold + 1}/{self.dataloader.k} progress: {seen}/{len(train_loader.dataset)}", end="\r")

        avg_fold_loss = fold_loss / len(train_loader.dataset)

        print(f"Fold {fold + 1}: Training loss = {avg_fold_loss :.4f}")
 
        self.writer.add_scalar(f"Loss/train_epoch (fold {fold})", avg_fold_loss, epoch)


    def evaluate(self, val_loader, epoch: int, fold: int) -> float:
        """Evaluate the model on the validation data."""
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        correct_top5 = 0 # Also track whether the model puts the correct option in its 'top 5 guesses'
        total = 0

        with torch.no_grad():  # No need to compute gradients during evaluation
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)  # Move data to the same device as the model
                outputs = self.model(images)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # Count correct predictions
                correct_top5 += (labels.unsqueeze(1) == torch.topk(outputs, 5, dim=1).indices).any(dim=1).sum().item() # 


        accuracy = correct / total if total > 0 else 0
        print(f"Fold {fold + 1}: Validation Accuracy: {accuracy:.4f}")

        top5_accuracy = correct_top5 / total if total > 0 else 0
        print(f"Fold {fold + 1}: Validation Top-5 Accuracy: {top5_accuracy:.4f}")

        self.writer.add_scalar(f"Accuracy/val (fold {fold})", accuracy, epoch)
        self.writer.add_scalar(f"Accuracy/val_top5 (fold {fold})", top5_accuracy, epoch)
        return accuracy



    def train(self, resume_from: str = None):
        """Train K different models for the specified number of epochs, with the option of resuming from a checkpoint."""
        for fold, train_loader, val_loader in self.dataloader.get_k_fold_dataloaders():
            print(f"Starting Fold {fold + 1}/{self.dataloader.k}")

            # When starting a new fold, we should reset all parameters to start fresh
            self.model.apply(
                lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None
            )
            self.model = self.model.to(self.device) # Ensure model is on the correct device after resetting parameters

            # We should also reset the optimizer and schedulerin case this carries any information from the previous fold
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # ← missing
                self.optimizer,
                mode='max',
                factor=self.config.training.scheduler.gamma,
                patience=self.config.training.scheduler.step_size,
            )

            start_epoch = 0

            if resume_from:
                start_epoch = self.load_checkpoint(resume_from)
                resume_from = None # Only resume from checkpoint for the first fold, after that we want to continue training without loading again

            epochs_no_improve = 0  # reset at the start of each fold

            for epoch in range(start_epoch, self.num_epochs):
                print(f"\n  Epoch {epoch + 1}/{self.num_epochs}")
 
                # Train for one epoch using this fold's training data
                self._train_single_epoch(train_loader, epoch, fold)
 
                # Evaluate on this fold's validation data
                val_accuracy = self.evaluate(val_loader, epoch=epoch, fold=fold)
                self.scheduler.step(val_accuracy) 

                # Log the current learning rate to TensorBoard
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar(f"LR/fold_{fold}", current_lr, epoch)

                # Early stopping
                if val_accuracy > self.best_val_accuracy:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.config.training.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
                
                self.save_checkpoint(epoch, val_accuracy)


        self.writer.close()
        print("Congratulations Congratulations Congratulations!")
        print("Training complete!")
 


if __name__ == "__main__":
    # Quick sanity check for the Trainer class
    with open("configs/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config = DictConfig(config) # Convert to DictConfig for consistency
    data = FoodDataset(config=config) # Use the config to initialize the dataset
    num_classes = len(data.labels_df['label'].unique()) # Dynamically determine number of classes from the dataset labels
    model = FoodClassifier(num_classes=num_classes, config=config)
    dataloader = FoodDataLoader(data, config=config)


    # Determine amount per class for weighted sampling
    # in train.py, after dataset is created
    label_counts = data.labels_df.iloc[:, 1].value_counts().sort_index()
    class_weights = (1.0 / label_counts).values
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)



    trainer = Trainer(model=model, dataloader=dataloader, config=config, class_weights=class_weights)
    trainer.train()
