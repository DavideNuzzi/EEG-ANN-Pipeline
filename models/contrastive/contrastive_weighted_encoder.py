import torch
import torch.nn as nn
from models.base_models import BaseModel


# Questo encoder usa direttamente il batch di dati, senza sample positivi e negativi
# e una funzione che dice come lavorare sui sample del batch per definirne la distanza
class EncoderContrastiveWeights(BaseModel):
    def __init__(self, layers: nn.Module, labels_distance, labels_weights, temperature=1, train_temperature=False):
        
        super(EncoderContrastiveWeights, self).__init__()

        self.layers = layers
        self.labels_distance = labels_distance
        self.train_temperature = train_temperature
        self.labels_weights = nn.Parameter(torch.tensor(labels_weights), requires_grad=False)

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
    def loss(self, f_x, weights):
        
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

        # Correggo i pesi per evitare di fare log(0) e ne prendo il logaritmo
        weights_log = torch.log(1e-6 + weights)

        # Costruisco un tensore in cui considero anche i pesi
        psi_weighted = psi.view((*psi.shape, 1)) + weights_log

        # Applico logsumexp ad entrambi e calcolo separatamente i due termini della loss
        # alignement = -torch.sum(torch.logsumexp(psi_weighted, dim=1))
        # uniformity = labels_num * torch.sum(torch.logsumexp(psi, dim=1))
        total_label_weight = self.labels_weights.sum()
        alignement = -torch.sum(self.labels_weights * torch.sum(torch.logsumexp(psi_weighted, dim=1), dim=0))
        uniformity = total_label_weight * torch.sum(torch.logsumexp(psi, dim=1))
        
        # psi_weighted_inv = psi.view((*psi.shape, 1)) + torch.log(1 - weights + 1e-10)
        # den = torch.sum(torch.logsumexp(psi_weighted_inv, dim=1))

        loss = alignement + uniformity

        return alignement, uniformity, loss
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, labels = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass       
        f_x = self.forward(x)

        # Pesi
        weights = self.labels_distance.get_weights(labels)

        # Loss
        alignement, uniformity, loss = self.loss(f_x, weights)

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'alignement': alignement.item(),
                'uniformity': uniformity.item(),
                'loss': loss.item()}

    