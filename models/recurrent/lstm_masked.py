import torch
import torch.nn as nn


# Impara una maschera sui tempi con cui scegliere i migliori 
# istanti da usare per la classificazione
class LSTMClassifierTimeMask(nn.Module):
    
    def __init__(self, input_size, input_times, hidden_size, num_layers, classifier: nn.Module, dropout_rate):
        
        super(LSTMClassifierTimeMask, self).__init__()

        self.hidden_size = hidden_size  # Numero di stati hidden per livello
        self.num_layers = num_layers    # Numero di livelli nella rete
        self.classifier = classifier    # Classificatore da usare
        
        self.attention_mask = nn.Parameter(torch.ones((1, input_times, 1)))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
    
    def forward(self, x):

        # Poiché mi aspetto che i dati siano (batch_size, 1, num_channels, num_timepoints)
        # ma la lstm si aspetta (batch_size, num_timepoints, num_channels), faccio una permutazione
        x = torch.permute(x.squeeze(), (0, 2, 1))
        
        # Passo attraverso la lstm e salvo gli stati hidden a ogni step
        hidden_states, _ = self.lstm(x)  # (batch_size, num_timepoints, hidden_size)

        # # Calcolo la versione pesata con l'attention mask
        hidden_states_merged = torch.sum(self.attention_mask *  hidden_states, dim=1) / torch.sum(self.attention_mask)

        # Ottengo la previsione (logit) delle classi
        y_logit = self.classifier(hidden_states_merged)
        
        return hidden_states, y_logit
    
    def loss(self, y, y_pred):
        return nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        _, y_logit = self.forward(x)

        # Loss
        loss = self.loss(y, y_logit)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        # Accuracy predizioni
        y_pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
        accuracy = torch.sum(y_pred_class == y)

        return {'loss': loss.item(), 'accuracy': accuracy.item()}
