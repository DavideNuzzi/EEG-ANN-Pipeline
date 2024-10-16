import torch
import torch.nn as nn


# Il classificatore viene applicato a ogni istante di tempo, quindi 
# a ogni hidden state, e la loss è anch'essa cumulativa
class LSTMClassifierAllTimes(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, classifier: nn.Module, classes_num, dropout_rate):
        
        super(LSTMClassifierAllTimes, self).__init__()

        self.hidden_size = hidden_size  # Numero di stati hidden per livello
        self.num_layers = num_layers    # Numero di livelli nella rete
        self.classifier = classifier    # Classificatore da usare
        self.classes_num = classes_num
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
    
    def forward(self, x):

        # Poiché mi aspetto che i dati siano (batch_size, 1, num_channels, num_timepoints)
        # ma la lstm si aspetta (batch_size, num_timepoints, num_channels), faccio una permutazione
        x = torch.permute(x.squeeze(), (0, 2, 1))
        
        _, num_timepoints, _ = x.shape

        # Passo attraverso la lstm e salvo gli stati hidden a ogni step
        hidden_states, _ = self.lstm(x)  # (batch_size, num_timepoints, hidden_size)
        
        # Ottengo una previsione della classe a partire da ogni hidden state
        y_logits = torch.zeros((x.shape[0], x.shape[1], self.classes_num), device=x.device)

        for i in range(num_timepoints):
            y_logits[:,i,:] = self.classifier(hidden_states[:, i, :])
        
        return hidden_states, y_logits
    
    def loss(self, y, y_pred):
        return nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        _, y_logits = self.forward(x)

        # Loss per tutti gli istanti
        # E anche accuratezza media e massima
        loss = 0
        accuracy_mean = 0.0
        accuracy_max = -1.0e10

        for i in range(y_logits.shape[1]):
            loss += self.loss(y, y_logits[:, i, :])

            y_pred_class = torch.argmax(torch.softmax(y_logits[:, i, :], dim=1), dim=1)
            accuracy = torch.sum(y_pred_class == y)    
            accuracy_mean += accuracy

            if accuracy > accuracy_max: accuracy_max = accuracy

        loss /= y_logits.shape[1]
        accuracy_mean /= y_logits.shape[1]

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'loss': loss.item(), 'accuracy_mean': accuracy_mean.item(), 'accuracy_max': accuracy_max.item()}
