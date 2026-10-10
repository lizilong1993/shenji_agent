import inspect

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


class MultiHorizonIntentionModel(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=128, horizons=5, num_classes=7):
        super().__init__()
        self.horizons = horizons
        self.num_classes = num_classes
        attention_kwargs = {"embed_dim": hidden_dim * 2, "num_heads": 4}
        self._attention_batch_first = "batch_first" in inspect.signature(nn.MultiheadAttention).parameters
        if self._attention_batch_first:
            attention_kwargs["batch_first"] = True
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.attention = nn.MultiheadAttention(**attention_kwargs)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, horizons * num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        if self._attention_batch_first:
            attn_out, _ = self.attention(out, out, out)
        else:
            out_t = out.transpose(0, 1)
            attn_out, _ = self.attention(out_t, out_t, out_t)
            attn_out = attn_out.transpose(0, 1)
        pooled = attn_out[:, -1, :]
        logits = self.head(pooled)
        return logits.view(-1, self.horizons, self.num_classes)


class TacticalScorer(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class LegacyIntentModel(nn.Module):
    """Load old IntentionLSTM checkpoints (fc_intent style) and output [batch, 1, 7].

    The restored checkpoint at ai/weights/intention_lstm.pth was trained with an
    8-class IntentionLSTM. This wrapper loads that checkpoint and adapts the output
    to match the current policy expectation: [batch, horizons, num_classes] = [batch, 1, 7].
    The 8th class (index 7) is truncated because _DIRECTION_STEPS only defines 0-6.
    """
    def __init__(self, input_dim=108, hidden_dim=256, output_classes=7):
        super().__init__()
        self.horizons = 1
        self.num_classes = output_classes
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4)
        self.fc_intent = nn.Linear(hidden_dim * 2, 8)  # 8 classes for checkpoint compatibility

    def forward(self, x):
        out, _ = self.lstm(x)
        out_t = out.transpose(0, 1)
        attn_out, _ = self.attention(out_t, out_t, out_t)
        attn_out = attn_out.transpose(0, 1)
        logits = self.fc_intent(attn_out[:, -1, :])      # [batch, 8]
        logits = logits[:, :self.num_classes]             # [batch, 7] - drop 8th class
        return logits.unsqueeze(1)                        # [batch, 1, 7]


class LegacySituationModel(nn.Module):
    """Load old SituationGCN checkpoints (bare keys) into GraphSituationModel-compatible interface.

    The restored checkpoint at ai/weights/situation_gcn.pth was saved from a bare
    SituationGCN with keys: sage1.linear.weight, fc_score.weight, etc. (no 'backbone.' prefix).
    This wrapper automatically prefixes bare keys on load.
    """
    def __init__(self, num_features=108, hidden_dim=64):
        super().__init__()
        self.inner = SituationGCN(num_features=num_features, hidden_dim=hidden_dim, output_dim=1)
        # Expose inner as `backbone` without registering as a submodule,
        # so policy.py line 567 (self.situation_model.backbone(x, adj)) works
        # without polluting the state dict with conflicting backbone.* keys.
        self.__dict__["backbone"] = self.inner

    def forward(self, x, adj):
        node_scores = self.inner(x, adj)
        return node_scores.mean(dim=0)

    def load_state_dict(self, state_dict, strict=True):
        """Override: auto-prefix bare keys with 'inner.' so bare SituationGCN checkpoints load."""
        is_bare = any(k.startswith("sage") or k.startswith("fc_") for k in state_dict)
        if is_bare:
            adjusted = {"inner." + k: v for k, v in state_dict.items()}
            return super().load_state_dict(adjusted, strict)
        return super().load_state_dict(state_dict, strict)


class GraphSituationModel(nn.Module):
    def __init__(self, num_features=108, hidden_dim=64):
        super().__init__()
        self.backbone = SituationGCN(num_features=num_features, hidden_dim=hidden_dim, output_dim=1)

    def forward(self, x, adj):
        node_scores = self.backbone(x, adj)
        return node_scores.mean(dim=0)


class ObjectiveModel(nn.Module):
    """objective 奖励回归器（I06 步骤 3）。

    由引擎裁决标签（`damage_kill_delta` / `occupy_success`）预测 `reward_proxy`，
    取代原先恒定的启发式打分（`objective_mode=heuristic_policy_score`）。

    放在 `ai/models.py` 而不是 `training/supervised.py`：
    `ai/policy.py` 已从本模块导入其余模型，放这里可避免 policy 反向依赖 training 层。
    """

    def __init__(self, num_features=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)
