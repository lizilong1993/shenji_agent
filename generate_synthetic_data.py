# Generate Synthetic Data for IntentionLSTM and SituationGCN
import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_data():
    base_dir = "datasets"
    os.makedirs(os.path.join(base_dir, "WG-StateGraph"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "WargameData_mini01"), exist_ok=True)

    print("Generating synthetic data...")
    
    # 1. Generate IntentionLSTM Data (Trajectories)
    # Format: Sequence of (x, y, v, theta) -> Intention Class (0-3)
    # 150,000 samples mentioned in report
    num_samples = 1000 # Use smaller number for demo
    seq_len = 10
    feature_dim = 108 # Model expects 108 features, though report says (x, y, v, theta) which is 4.
                      # Let's align with the model definition in ai/models.py: input_dim=108
    
    for i in range(5): # Generate a few files
        data = []
        for _ in range(200):
            # Random features: [seq_len, feature_dim]
            features = np.random.randn(seq_len, feature_dim).astype(np.float32)
            # Random label: 0-3
            label = np.random.randint(0, 4)
            data.append({"features": features, "label": label})
        
        save_path = os.path.join(base_dir, "WargameData_mini01", f"traj_data_{i}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {save_path}")

    # 2. Generate SituationGCN Data (Graphs)
    # Format: Node features (N, 108), Adjacency Matrix (N, N) -> Score (0-1)
    # 24,080 samples mentioned in report
    
    for i in range(5):
        data = []
        for _ in range(100):
            num_nodes = np.random.randint(5, 20)
            # Node features
            x = np.random.randn(num_nodes, 108).astype(np.float32)
            # Adjacency matrix (symmetric, binary for now)
            adj = np.random.randint(0, 2, (num_nodes, num_nodes)).astype(np.float32)
            # Make symmetric
            adj = (adj + adj.T) > 0
            adj = adj.astype(np.float32)
            # Score
            score = np.random.rand()
            
            data.append({"x": x, "adj": adj, "y": score})
            
        save_path = os.path.join(base_dir, "WG-StateGraph", f"graph_data_{i}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    generate_data()
