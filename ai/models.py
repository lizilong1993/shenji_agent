import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentionLSTM(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=256, num_classes=4):
        super(IntentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4)
        self.fc_intent = nn.Linear(hidden_dim*2, num_classes)  # ATTACK, DEFENSE, RECON, RETREAT

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # output shape: [batch, seq_len, hidden_dim*2]
        out, _ = self.lstm(x)  
        # Attention requires [seq_len, batch, embed_dim], so transpose
        out_t = out.transpose(0, 1)
        attn_out, _ = self.attention(out_t, out_t, out_t)
        # Transpose back: [batch, seq_len, embed_dim]
        attn_out = attn_out.transpose(0, 1)
        # Use the last time step for classification
        return self.fc_intent(attn_out[:, -1, :])

class GraphSageLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSageLayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, adj):
        # x: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes] (adjacency matrix)
        
        # Aggregate neighbors (mean aggregation)
        # Using simplified aggregation: neighbors = adj * x
        neighbors = torch.mm(adj, x)
        degree = adj.sum(dim=1, keepdim=True) + 1e-6
        neighbors = neighbors / degree
        
        # Concatenate self features with aggregated neighbor features
        combined = torch.cat([x, neighbors], dim=1)
        
        # Apply linear transformation and activation
        return F.relu(self.linear(combined))

class SituationGCN(nn.Module):
    def __init__(self, num_features=108, hidden_dim=64, output_dim=1):
        super(SituationGCN, self).__init__()
        self.sage1 = GraphSageLayer(num_features, hidden_dim)
        self.sage2 = GraphSageLayer(hidden_dim, hidden_dim)
        self.sage3 = GraphSageLayer(hidden_dim, hidden_dim)
        self.fc_score = nn.Linear(hidden_dim, output_dim) # Situation Score / Threat Level

    def forward(self, x, adj):
        x = self.sage1(x, adj)
        x = self.sage2(x, adj)
        x = self.sage3(x, adj)
        return torch.sigmoid(self.fc_score(x))
