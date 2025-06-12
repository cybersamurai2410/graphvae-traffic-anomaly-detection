# Runs the trained GraphVAE model on real traffic data (METR-LA)
# Outputs per-window reconstruction error and visual anomaly score plot

import torch
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MetrLaDataset
from graph_vae import GraphVAE

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed METR-LA dataset as batches of 1 sequence
dataset = MetrLaDataset("data/METR-LA", seq_len=12)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Define model and load trained weights
num_nodes = dataset[0][0].shape[0]
model = GraphVAE(in_channels=12, hidden_dim=64, latent_dim=32, num_nodes=num_nodes)
model.load_state_dict(torch.load("graph_vae_metrla.pt", map_location=device))
model.to(device)
model.eval()

# Evaluate reconstruction error per window
indices = []
scores = []

with torch.no_grad():
    for i, (x, edge_index) in enumerate(loader):
        x = x.to(device)                     # shape: [1, N, T]
        edge_index = edge_index.to(device)  # shape: [2, num_edges]

        # Forward pass through the trained VAE
        recon, mu, logvar = model(x, edge_index)

        # Calculate average reconstruction error
        error = (recon - x).pow(2).mean().item()
        indices.append(i)
        scores.append(error)

# Write anomaly scores to CSV file
with open("anomaly_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["window_index", "reconstruction_error"])
    writer.writerows(zip(indices, scores))

print("Saved anomaly_scores.csv")

# Create and save line plot of anomaly scores
plt.plot(indices, scores)
plt.xlabel("Window Index")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Scores (GraphVAE)")
plt.savefig("anomaly_scores.png")
plt.close()
print("Saved anomaly_scores.png")
