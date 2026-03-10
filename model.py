import torch
import torch.nn as nn

# Define convolutional neural network for image classification
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        