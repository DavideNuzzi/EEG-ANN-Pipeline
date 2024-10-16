import torch
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


class BaseClassifier(BaseModel):

    def loss(self, y, y_pred):
        return nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        y_pred = self.forward(x)

        # Loss
        loss = self.loss(y, y_pred)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        # Accuracy predizioni
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        accuracy = torch.sum(y_pred_class == y)

        return {'loss': loss.item(), 'accuracy': accuracy.item()}

    def predict(self, x):
        
        # Forward pass
        y_pred = self(x)

        # Faccio il softmax e poi prendo l'indice con probabilità maggiore
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        return y_pred_class