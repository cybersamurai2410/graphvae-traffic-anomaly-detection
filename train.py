# Trains the GraphVAE model on the METR-LA dataset and saves the model and loss plot.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MetrLaDataset
from graph_vae import GraphVAE

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the traffic dataset and define a DataLoader
dataset = MetrLaDataset("data/METR-LA", seq_len=12)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get the number of nodes (sensors)
num_nodes = dataset[0][0].shape[0]

# Initialize GraphVAE model
model = GraphVAE(in_channels=12, hidden_dim=64, latent_dim=32, num_nodes=num_nodes)
model.to(device)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Tracks average loss per epoch for plotting
loss_history = []

# Loss function: combines reconstruction MSE and KL divergence
def compute_loss(recon_x, x, mu, logvar, beta=1.0):
    mse = F.mse_loss(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld

# Train for 20 epochs
for epoch in range(1, 21):
    model.train()
    total_loss = 0.0

    for x, edge_index in loader:
        x = x.to(device)
        edge_index = edge_index.to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, edge_index)
        loss = compute_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

# Save model weights
torch.save(model.state_dict(), "graph_vae_metrla.pt")
print("Saved model to graph_vae_metrla.pt")

# Plot and save training loss graph
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GraphVAE Training Loss")
plt.savefig("training_loss.png")
plt.close()
print("Saved plot to training_loss.png")
