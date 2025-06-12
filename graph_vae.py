# Defines the encoder, decoder, and full GraphVAE model.
# Encoder uses GCN layers to produce latent mean and log-variance.
# Decoder reconstructs full node features from the latent vector.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.mean(dim=0)  # Global mean pooling
        return self.fc_mu(x), self.fc_logvar(x)


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_channels, num_nodes):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_nodes * out_channels)
        self.num_nodes = num_nodes
        self.out_channels = out_channels

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = self.fc2(h)
        return out.view(self.num_nodes, self.out_channels)

class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, num_nodes):
        super().__init__()
        self.encoder = GraphEncoder(in_channels, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, in_channels, num_nodes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
