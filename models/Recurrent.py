import torch
import torch.nn as nn



class LSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, classifier: nn.Module, dropout_rate):
        
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size  # Numero di stati hidden per livello
        self.num_layers = num_layers    # Numero di livelli nella rete
        self.classifier = classifier    # Classificatore da usare
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
    
    def forward(self, x):

        # Poiché mi aspetto che i dati siano (batch_size, 1, num_channels, num_timepoints)
        # ma la lstm si aspetta (batch_size, num_timepoints, num_channels), faccio una permutazione
        x = torch.permute(x.squeeze(), (0, 2, 1))
        
        # Passo attraverso la lstm e salvo gli stati hidden a ogni step
        hidden_states, _ = self.lstm(x)  # (batch_size, num_timepoints, hidden_size)
        
        # Prendo l'ultimo hidden state
        final_hidden_state = hidden_states[:, -1, :] 
        
        # Ottengo la previsione (logit) delle classi
        y_logit = self.classifier(final_hidden_state)
        
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


# Impara una maschera di attenzione con cui scegliere i migliori 
# istanti di tempo da usare per la classificazione
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



# Usa un meccanismo di attenzione per capire su quali istanti di tempo
# concentrarsi e produrre la migliore rappresentazione per la classificazione
class LSTMClassifierAttention(nn.Module):
    
    def __init__(self, input_size, input_times, hidden_size, num_layers, classifier: nn.Module, dropout_rate):
        
        super(LSTMClassifierAttention, self).__init__()

        self.hidden_size = hidden_size  # Numero di stati hidden per livello
        self.num_layers = num_layers    # Numero di livelli nella rete
        self.classifier = classifier    # Classificatore da usare
        
        self.embed_size = hidden_size
        self.q_lin = nn.Linear(self.hidden_size, 1)
        self.k_lin = nn.Linear(self.hidden_size, 1)
        # self.q_conv = nn.Conv1d(self.hidden_size, 1, kernel_size=16, padding='same')
        # self.k_conv = nn.Conv1d(self.hidden_size, 1, kernel_size=16, padding='same')
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)

        # self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.5, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
    
    def forward(self, x):

        # Poiché mi aspetto che i dati siano (batch_size, 1, num_channels, num_timepoints)
        # ma la lstm si aspetta (batch_size, num_timepoints, num_channels), faccio una permutazione
        x = torch.permute(x.squeeze(), (0, 2, 1))
        
        # Passo attraverso la lstm e salvo gli stati hidden a ogni step
        hidden_states, _ = self.lstm(x)  # (batch_size, num_timepoints, hidden_size)

        # # Uso l'attention layer per produrre un embedding con attention
        # embedding, _ = self.attention(hidden_states, hidden_states, hidden_states) 
        # Mi interessa solo l'ultimo istante
        # embedding_final = embedding[:, -1, :]

        # hidden_states_reshaped = torch.permute(hidden_states, (0, 2, 1))
        # query = self.dropout_1(self.q_conv(hidden_states_reshaped))
        # key = self.dropout_2(self.k_conv(hidden_states_reshaped))
        # attention_scores = torch.softmax(query * key, dim=2)
        # attention_scores = torch.permute(attention_scores, (0, 2, 1))

        query = self.dropout_1(self.q_lin(hidden_states))
        key = self.dropout_2(self.k_lin(hidden_states))
        attention_scores = torch.softmax(query * key / (self.hidden_size ** 0.5), dim=1)
        
        values = hidden_states * attention_scores

        embedding_final = torch.sum(values, dim=1)

        # Ottengo la previsione (logit) delle classi
        y_logit = self.classifier(embedding_final)
        
        return hidden_states, attention_scores, y_logit
    
    def loss(self, y, y_pred):
        return nn.functional.cross_entropy(y_pred, y, reduction='mean') * y.shape[0]
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        _, _, y_logit = self.forward(x)

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
