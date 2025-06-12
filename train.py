# Trains the GraphVAE model on the METR-LA dataset.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MetrLaDataset
from graph_vae import GraphVAE

def compute_loss(recon_x, x, mu, logvar, beta=1.0):
    mse = F.mse_loss(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MetrLaDataset("data/METR-LA", seq_len=12)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_nodes = dataset[0][0].shape[0]
    model = GraphVAE(12, 64, 32, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for x, edge_index in loader:
            x = x.to(device)
            edge_index = edge_index.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, edge_index)
            loss = compute_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        loss_history.append(avg)
        print(f"Epoch {epoch}, Loss: {avg:.6f}")

    torch.save(model.state_dict(), "graph_vae_metrla.pt")

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GraphVAE Training Loss")
    plt.savefig("training_loss.png")
    plt.close()

if __name__ == "__main__":
    main()
