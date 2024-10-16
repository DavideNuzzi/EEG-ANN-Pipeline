import torch
import torch.nn as nn
from models.base_models import BaseModel


class VAE(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super(VAE, self).__init__()
        self.encoder = encoder

        # Use LazyLinear to defer determining the input size until the first forward pass
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)
        self.decoder = decoder

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
        return decoded, mu, logvar

    def loss(self, x, reconstructed, mu, logvar):

        reconstruction_loss = torch.mean((reconstructed - x)**2) * x.shape[0]
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * x.shape[0]

        return reconstruction_loss, kl_divergence

    def process_batch(self, batch, optimizer, is_eval=False):
        
        x, y = batch

        # Resetto i gradienti
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass
        reconstructed, mu, logvar = self(x)

        # Loss
        reconstruction_loss, kl_divergence = self.loss(x, reconstructed, mu, logvar)
        loss = reconstruction_loss + kl_divergence

        # Backpropagation
        if not is_eval: 
            loss.backward()
            optimizer.step()

        return {'reconstruction_loss': reconstruction_loss.item(),
                'kl_loss': kl_divergence.item(), 
                'loss': loss.item()}
    