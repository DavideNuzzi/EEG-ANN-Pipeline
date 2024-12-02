import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from metrics.metrics import MeanMetric, ClassificationMetric
from metrics import Accuracy

class BaseModel(nn.Module, ABC):

    def __init__(self):
        super(BaseModel, self).__init__()

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

    def __init__(self, metrics=None, loss_weights=None):
        super(BaseModel, self).__init__()

        if metrics is None:
            metrics = {'loss': MeanMetric(), 'accuracy': Accuracy()}
        
        self.metrics = metrics
        self.loss_weights = loss_weights

    def loss(self, y, y_pred):

        if self.loss_weights is not None:
            return nn.functional.cross_entropy(y_pred, y, reduction='mean', weight=self.loss_weights)
        else:
            return nn.functional.cross_entropy(y_pred, y, reduction='mean')
    
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

        # Classe predetta
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        # Passo le informazioni utili alla funzione che aggiorna le metriche
        # Questa funzione dovrà essere definita dalla classe specifica
        if self.metrics is not None:
            return self.update_metrics(loss, y, y_pred_class)
        else:
            # Accuracy predizioni
            accuracy = torch.sum(y_pred_class == y)

            return {'loss': loss.item(), 'accuracy': accuracy.item()}

    def update_metrics(self, loss, y, y_pred):

        result = dict()

        for metric_name in self.metrics:

            metric = self.metrics[metric_name]

            if isinstance(metric, MeanMetric):
                metric.update(loss)
                result[metric_name] = metric.result()
                
            elif isinstance(metric, ClassificationMetric):
                metric.update(y, y_pred)
                result[metric_name] = metric.result()

        return result

    def predict(self, x):
        
        # Forward pass
        y_pred = self(x)

        # Faccio il softmax e poi prendo l'indice con probabilità maggiore
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        return y_pred_class
    
