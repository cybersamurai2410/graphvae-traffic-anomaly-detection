# Loads and processes the METR-LA dataset into (node_features, edge_index) samples for unsupervised training of a Graph Variational Autoencoder.

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler

class MetrLaDataset(Dataset):
    """
    Dataset that loads METR-LA traffic speed data, builds time-windowed node features,
    and provides the graph edge index for GNN models.
    """
    def __init__(self, data_dir="data/METR-LA", seq_len=12):
        self.seq_len = seq_len

        # Load adjacency matrix (used for GNN edge connections)
        with open(os.path.join(data_dir, "adj_mx.pkl"), "rb") as f:
            _, _, adj_mx = pickle.load(f, encoding="latin1")
        self.edge_index, _ = dense_to_sparse(torch.tensor(adj_mx, dtype=torch.float32))

        # Load traffic speed data (shape: time x 207 sensors)
        df = pd.read_hdf(os.path.join(data_dir, "metr-la.h5"))
        values = df.values.astype(np.float32)

        # Normalize across time for each sensor
        scaler = StandardScaler()
        values = scaler.fit_transform(values)

        # Convert to windowed sequences: (num_nodes, seq_len)
        self.samples = []
        for i in range(len(values) - seq_len):
            window = values[i:i + seq_len].T
            self.samples.append(torch.tensor(window, dtype=torch.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.edge_index
