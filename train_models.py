import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ai.models import IntentionLSTM, SituationGCN

# ---------------------------------------------------------
# Datasets
# ---------------------------------------------------------

class IntentionDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pkl"):
                    with open(os.path.join(data_dir, file), 'rb') as f:
                        self.samples.extend(pickle.load(f))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        # features: [seq_len, feature_dim]
        # label: int
        return torch.tensor(item['features'], dtype=torch.float32), torch.tensor(item['label'], dtype=torch.long)

class SituationDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pkl"):
                    with open(os.path.join(data_dir, file), 'rb') as f:
                        self.samples.extend(pickle.load(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # x: [num_nodes, features]
        # adj: [num_nodes, num_nodes]
        # y: float
        return (torch.tensor(item['x'], dtype=torch.float32), 
                torch.tensor(item['adj'], dtype=torch.float32), 
                torch.tensor(item['y'], dtype=torch.float32))

# ---------------------------------------------------------
# Training Functions
# ---------------------------------------------------------

def train_intention_lstm(data_dir, output_dir, epochs=5):
    print(f"\n--- Training IntentionLSTM ---")
    dataset = IntentionDataset(data_dir)
    if len(dataset) == 0:
        print("No data found for IntentionLSTM.")
        return
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model config from models.py: input_dim=108, hidden_dim=256, num_classes=4
    # The synthetic data matches feature_dim=108.
    model = IntentionLSTM(input_dim=108, hidden_dim=256, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "intention_lstm.pth"))
    print(f"Model saved to {output_dir}/intention_lstm.pth")


def train_situation_gcn(data_dir, output_dir, epochs=5):
    print(f"\n--- Training SituationGCN ---")
    dataset = SituationDataset(data_dir)
    if len(dataset) == 0:
        print("No data found for SituationGCN.")
        return
    
    # Custom collate function for variable size graphs (batching not trivial without PyG)
    # For simplicity, batch_size=1 here or implement padding.
    # Let's use batch_size=1 for standard PyTorch GCN demo.
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = SituationGCN(num_features=108, hidden_dim=64, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, adj, y) in enumerate(loader):
            # x: [1, N, F], adj: [1, N, N], y: [1]
            x = x.squeeze(0)
            adj = adj.squeeze(0)
            y = y.view(-1, 1) # [1, 1]
            
            optimizer.zero_grad()
            output = model(x, adj) # Output [1, 1] (or [N, 1] if node level? No, global score)
            
            # SituationGCN.forward returns [1, 1]
            # Wait, check models.py:
            # return torch.sigmoid(self.fc_score(x)) -> shape depends on pooling
            # In models.py: x = self.sage3(x, adj) -> [N, hidden]
            # self.fc_score(x) -> [N, 1]
            # So it returns scores per node?
            # Re-checking models.py...
            
            # Correction: SituationGCN output in models.py is likely node-level if no pooling.
            # Let's check models.py content again.
            # class SituationGCN(nn.Module): ... return torch.sigmoid(self.fc_score(x))
            # Yes, it returns [N, 1].
            # But we want a global score.
            # Let's add global pooling (mean) here or assume labels are per-node?
            # The synthetic data has 1 label per graph.
            # So we should mean-pool the output.
            
            node_scores = model(x, adj)
            graph_score = torch.mean(node_scores, dim=0) # [1]
            
            loss = criterion(graph_score, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "situation_gcn.pth"))
    print(f"Model saved to {output_dir}/situation_gcn.pth")

if __name__ == "__main__":
    train_intention_lstm("datasets/WargameData_mini01", "ai/weights")
    train_situation_gcn("datasets/WG-StateGraph", "ai/weights")
