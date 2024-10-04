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

  