import torch
import torch.nn as nn
from models.base_models import BaseModel


class VAEClassifier(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, classifier: nn.Module, latent_dim: int, num_classes: int):
        super(VAEClassifier, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        # Use LazyLinear to defer determining the input size until the first forward pass
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_flatten = encoded.view(x.size(0), -1)  # Flatten the encoded output
        mu = self.fc_mu(encoded_flatten)
        logvar = self.fc_logvar(encoded_flatten)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        y_pred = self.classifier(z)

        return decoded, mu, logvar, y_pred

    def loss(self, x, y, reconstructed, mu, logvar, y_pred):

        reconstruction_loss = torch.mean((reconstructed - x)**2) * x.shape[0] 
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * x.shape[0]
        classifier_loss = torch.nn.functional.cross_entropy(y_pred, y, reduction='mean') * x.shape[0]

        return reconstruction_loss, kl_divergence, classifier_loss

    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        reconstructed, mu, logvar, y_pred = self.forward(x)

        # Loss
        reconstruction_loss, kl_divergence, classifier_loss = self.loss(x, y, reconstructed, mu, logvar, y_pred)
        loss = reconstruction_loss + kl_divergence + classifier_loss

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'reconstruction_loss': reconstruction_loss.item(),
                'kl_loss': kl_divergence.item(), 
                'classifier_loss': classifier_loss.item(),
                'loss': loss.item()}
    
    def predict(self, x):
        
        # Forward pass
        _, _, _, y_pred = self(x)

        # Faccio il softmax e poi prendo l'indice con probabilità maggiore
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        return y_pred_class

