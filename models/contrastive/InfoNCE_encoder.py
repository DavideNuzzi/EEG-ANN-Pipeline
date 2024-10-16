import torch
import torch.nn as nn
from models.base_models import BaseModel

# Encoder come quello di CEBRA, ogni batch di dati contiene N sample di riferimento
# N sample positivi (uno per ogni riferimento) e N sample negativi (N per ogni riferimento,
# condivisi tra tutti).
class EncoderInfoNCE(BaseModel):
    def __init__(self, layers: nn.Module, temperature: float, train_temperature=False):
        
        super(EncoderInfoNCE, self).__init__()

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
        f_y_neg = self.forward(y_neg)

        # Caso single-label  
        if type(y_pos) is not dict:
            
            f_y_pos = self.forward(y_pos)
            alignement, uniformity, loss = self.loss(f_x, f_y_pos, f_y_neg)
        else:

            # Caso multi-label
            loss = 0
            alignement = 0
            uniformity = 0

            for label_name in y_pos:
                y_pos_i = y_pos[label_name]
                f_y_pos = self.forward(y_pos_i)

                # Loss
                alignement_i, uniformity_i, loss_i = self.loss(f_x, f_y_pos, f_y_neg)
                alignement += alignement_i
                uniformity += uniformity_i
                loss += loss_i

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'alignement': alignement.item(),
                'uniformity': uniformity.item(),
                'loss': loss.item()}
