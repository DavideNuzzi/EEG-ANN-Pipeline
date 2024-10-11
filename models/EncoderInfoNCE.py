import torch
import torch.nn as nn
from models.BaseModel import BaseModel


class EncoderInfoNCE(BaseModel):
    def __init__(self, layers: nn.Module, temperature: float, normalize_latents=False):
        
        super(EncoderInfoNCE, self).__init__()

        self.layers = layers
        self.temperature = temperature
        self.normalize_latents = normalize_latents

    def forward(self, x, y_pos, y_neg):

        # x e y_pos vengono passati nell'encoder così come sono
        f_x = self.layers(x)
        f_y_pos = self.layers(y_pos)

        # y_neg ha dimensione (batch, 10, 1, channels, times)
        # Per rendere più veloce il calcolo metto assieme le dimensioni
        # del batch e dei vari sample, le processo assieme e poi rimetto
        # il tensore nella forma corretta nel return
        f_y_neg = self.layers(y_neg.view(-1, 1, x.shape[2], x.shape[3]))

        # Se devo normalizzare i vettori latenti (su una sfera)
        if self.normalize_latents:
            f_x = torch.nn.functional.normalize(f_x, dim=1)
            f_y_pos = torch.nn.functional.normalize(f_y_pos, dim=1)
            f_y_neg = torch.nn.functional.normalize(f_y_neg, dim=1)

        return f_x, f_y_pos, f_y_neg.view(x.shape[0], y_neg.shape[1], -1)

    def loss(self, f_x, f_y_pos, f_y_neg):

        # Calcolo le varie similarità tra sample positivi e negativi
        psi_pos = self.get_similarity(f_x, f_y_pos)

        # Per il termine negativo devo sommare separatamente sui samples 
        # e poi alla fine fare il logaritmo
        neg_term = 0
        for i in range(f_y_neg.shape[1]):
            neg_term += torch.exp(self.get_similarity(f_x, f_y_neg[:,i,:]))
        neg_term = torch.log(neg_term)
        
        # Ora entrambi i termini sono di dimensione (batch, 1), quindi li sommo e medio
        loss = torch.mean(-psi_pos + neg_term)
        
        return loss
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y_pos, y_neg, _ = batch
            
        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        f_x, f_y_pos, f_y_neg = self.forward(x, y_pos, y_neg)

        # Loss
        loss = self.loss(f_x, f_y_pos, f_y_neg)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'loss': loss.item()}

    # Implementa la funzione psi(x,y)
    def get_similarity(self, x, y):

        # Se devo normalizzare i vettori latenti la misura di similarità 
        # è il prodotto scalare, altrimenti la norma euclidea.
        # Ricordo che x e y hanno dimensioni (batch, latent_dim)
        # Il risultato ha dimensione (batch, 1)
        if self.normalize_latents:
            # Prodotto scalare
            return torch.sum(x * y, dim=1) / self.temperature
        else:
            # Norma euclidea standard
            return -torch.norm(x - y, p=2, dim=1) / self.temperature

  



# Questo encoder usa direttamente il batch di dati, senza sample positivi e negativi
# e una funzione che dice come lavorare sui sample del batch per definirne la distanza
class EncoderContrastive(BaseModel):
    def __init__(self, layers: nn.Module, temperature: float, train_temperature=False):
        
        super(EncoderContrastive, self).__init__()

        self.layers = layers
        self.train_temperature = train_temperature

        if train_temperature:
            self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            self.temperature_min = torch.tensor(0.0001)
        else:
            self.temperature = torch.tensor(temperature)

    def forward(self, x):

        # Encoding di tutto il batch
        f_x = self.layers(x)

        # Normalizzo
        f_x = torch.nn.functional.normalize(f_x, dim=1, p=2)

        return f_x

    # La loss in questo caso riceve solo f_x, poiché deve poi prendere ogni sample
    # del batch come riferimento e tutti gli altri come esempi positivi/negativi
    def loss(self, f_x, f_y_pos, f_y_neg):
    
        # Temperatura fissa?
        if self.train_temperature:
            temperature = torch.min(torch.exp(self.temperature), 1/self.temperature_min)
        else:
            temperature = self.temperature

        # Similarità
        psi_pos = torch.einsum('ai,ai->a', f_x, f_y_pos) / temperature
        psi_neg = torch.einsum('ai,bi->ab', f_x, f_y_neg) / temperature

        # Correzione per stabilità
        with torch.no_grad():
            c, _ = psi_neg.max(dim=1)

        psi_pos -= c.detach()
        psi_neg -= c.detach()
        
        loss_pos = -psi_pos.mean()
        loss_neg = torch.logsumexp(psi_neg, dim=1).mean()

        loss = loss_pos + loss_neg

        return loss_pos, loss_neg, loss
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y_pos, y_neg, _ = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass       
        f_x = self.forward(x)
        f_y_pos = self.forward(y_pos)
        f_y_neg = self.forward(y_neg)

        # Loss
        alignement, uniformity, loss = self.loss(f_x, f_y_pos, f_y_neg)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'alignement': alignement.item(),
                'uniformity': uniformity.item(),
                'loss': loss.item()}
















# Questo encoder usa direttamente il batch di dati, senza sample positivi e negativi
# e una funzione che dice come lavorare sui sample del batch per definirne la distanza
class EncoderInfoNCEW(BaseModel):
    def __init__(self, layers: nn.Module, temperature: float, label_weights,  train_temperature=False):
        
        super(EncoderInfoNCEW, self).__init__()

        self.layers = layers
        self.train_temperature = train_temperature

        if train_temperature:
            self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            self.temperature_min = torch.tensor(0.0001)
        else:
            self.temperature = torch.tensor(temperature)

        self.label_weights = label_weights

    def forward(self, x):

        # Encoding di tutto il batch
        f_x = self.layers(x)

        # Normalizzo
        f_x = torch.nn.functional.normalize(f_x, dim=1, p=2)

        return f_x

    # La loss in questo caso riceve solo f_x, poiché deve poi prendere ogni sample
    # del batch come riferimento e tutti gli altri come esempi positivi/negativi
    def loss(self, f_x, weights):
        
        if weights.ndim == 2:   # Single-label
            labels_num = 1
        else:                   # Multi-label
            labels_num = weights.shape[2]

        # Temperatura fissa?
        if self.train_temperature:
            temperature = torch.min(torch.exp(self.temperature), 1/self.temperature_min)
        else:
            temperature = self.temperature

        # Costruisco un tensore con tutte le similarità (in prodotto scalare)
        psi = torch.einsum('ai,bi->ab', f_x, f_x) / temperature

        # Correggo gli elementi sulla diagonale in modo che vengano scartati poi
        # dalle sommatorie
        psi.fill_diagonal_(-1e15)
        # Correzzione più forte
        # psi = psi.triu(diagonal=1) + (torch.ones(*psi.shape, device=psi.device) - 1e10).tril()
        
        # Correggo i pesi per evitare di fare log(0) e ne prendo il logaritmo
        weights_log = torch.log(1e-6 + weights)

        # Costruisco un tensore in cui considero anche i pesi
        psi_weighted = psi.view((*psi.shape, 1)) + weights_log

        # Applico logsumexp ad entrambi e calcolo separatamente i due termini della loss
        num = -torch.sum(torch.logsumexp(psi_weighted, dim=1))
        den = labels_num * torch.sum(torch.logsumexp(psi, dim=1))
        
        # psi_weighted_inv = psi.view((*psi.shape, 1)) + torch.log(1 - weights + 1e-10)
        # den = torch.sum(torch.logsumexp(psi_weighted_inv, dim=1))


        # Calcolo la loss complessiva (con un peso maggiore all'uniformità)
        loss = num + den

        return loss
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, labels, idx = batch

        batch_size = x.shape[0]

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass       
        f_x = self.forward(x)

        # Pesi
        # weights = self.label_weights[idx, :, :][:, idx, :]

        # Calcolo i pesi confrontando tutte le label
        if type(labels) is dict:
            weights = torch.zeros((batch_size, batch_size, len(labels)), device=x.device)
        else:
            weights = torch.zeros((batch_size, batch_size, 1), device=x.device)

        for i in range(batch_size):
            lab = {key: labels[key][i] for key in labels}
            weights[i, :, :] = self.label_weights(lab, labels)

        # Loss
        loss = self.loss(f_x, weights)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'loss': loss.item()}
