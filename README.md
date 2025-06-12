# Anomaly Detection in Urban Traffic using Graph Variational Autoencoders

This project uses a Graph Variational Autoencoder (GraphVAE) to detect unusual traffic patterns using real-world road sensor data from the METR-LA dataset. The model is designed to learn what “normal” traffic flow looks like and flag deviations using reconstruction error.

A graph neural network (GCN) is used in the encoder to learn spatial dependencies between sensors. The latent representation is sampled from a Gaussian distribution using variational reparameterization, and a fully connected decoder reconstructs the original input. High reconstruction error indicates a potential anomaly such as a traffic jam, incident, or unexpected congestion.

---

## Dataset

**METR-LA** is a publicly available traffic dataset collected by the Los Angeles County transportation system. It includes:

* **207 sensors (nodes)** installed on roads across LA
* **5-minute interval recordings**
* **Traffic speed (in mph)** as the main feature
* **Total duration**: several months of data
* **Adjacency matrix** representing sensor-to-sensor road connections

### Raw Data Format

* `metr-la.h5`: a matrix of shape **\[Timestamps × 207]**, where each row is a time point and each column is a road sensor.
* `adj_mx.pkl`: a **\[207 × 207]** adjacency matrix that defines physical road connectivity between sensors.

---

## Preprocessing

1. **Time-window slicing**

   * For each time step, the last **12 time points** (1 hour) are taken as input.
   * Each sample is shaped: **\[207 nodes × 12 features]**

2. **Graph construction**

   * The adjacency matrix is converted into a sparse format (`edge_index`) to be used by the GCN encoder.

3. **Normalization**

   * Each sensor’s speed data is standardized using `StandardScaler()` (zero mean, unit variance) to help the model learn efficiently.

---

## Model Architecture

* **Encoder**: 2-layer GCN to aggregate traffic features from neighboring sensors
* **Latent space**: Gaussian vector with learned μ and σ² per input
* **Decoder**: 2-layer MLP to reconstruct all sensor readings for the input window
* **Loss**: combines mean squared reconstruction loss and KL divergence

---

## Training Results

The model was trained for 20 epochs using the Adam optimizer. The training loss decreases smoothly and stabilizes:

![training_loss](https://github.com/user-attachments/assets/b6969035-4f9d-4803-89c4-22cedda8ad19)

---

## Evaluation

During inference, the model is evaluated on each window and its reconstruction error is computed. High error indicates abnormal traffic behavior not seen during training.

Anomaly scores are visualized below. The red threshold marks likely anomalous windows (accidents, congestion spikes, sensor faults):

![anomaly_scores](https://github.com/user-attachments/assets/1f62c962-f855-4ff4-89d1-32f3427e0320)

