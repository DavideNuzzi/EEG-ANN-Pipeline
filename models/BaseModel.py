import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, *args):
        """Defines the forward pass for the model."""
        pass

    @abstractmethod
    def loss(self, *args):
        """Specifies how to calculate the loss for this model"""
        pass

    @abstractmethod
    def process_batch(self, batch, optimizer, is_eval):
        """Processes a batch for either training or evaluation. 
        It should implement the backpropagation step and return
        all the metrics that you need to keep track of"""
        pass
