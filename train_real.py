# Real Data Training Script
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from ai.models import IntentionLSTM, SituationGCN
import glob

# ---------------------------------------------------------
# Intention Recognition Dataset (WargameData_mini01)
# ---------------------------------------------------------

class RealIntentionDataset(Dataset):
    def __init__(self, json_file):
        print(f"Loading {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.samples = []
        # The data is a list of lists of frames? Or a list of games?
        # Based on snippet: [ [ {operators: ...}, ... ] ]
        # It's a list of games, where each game is a list of frames.
        # Or a list of frames, where each frame is a list?
        # Snippet: `[ [ { "operators": ...`
        # Looks like: Outer List -> List (Game?) -> Dict (Frame)
        # So raw_data is List[List[Frame]]
        
        for game in self.raw_data:
            for frame in game:
                if not isinstance(frame, dict):
                    continue
                
                if not frame.get('tank_enemy'):
                    continue
            
            tank_info = frame['tank_enemy'][0]
            tank_id = tank_info[0]
            label_vec = tank_info[2] # First 20 steps
            
            # Find tank in operators
            tank_op = None
            for op in frame['operators']:
                if op['obj_id'] == tank_id:
                    tank_op = op
                    break
            
            if not tank_op:
                continue
                
            # Construct Features
            # Model expects [seq_len, 108]
            # Since we only have a snapshot, we will replicate the snapshot or use dummy history?
            # Or maybe 'move_path' can serve as a sequence?
            # Let's use a single step sequence for now: seq_len=1
            # Feature extraction: [x, y, speed, blood, ...] -> pad to 108
            
            feats = np.zeros(108, dtype=np.float32)
            feats[0] = tank_op.get('cur_hex', 0) / 10000.0 # Normalize roughly
            feats[1] = tank_op.get('cur_pos', 0)
            feats[2] = tank_op.get('speed', 0)
            feats[3] = tank_op.get('blood', 0)
            feats[4] = tank_op.get('max_blood', 1)
            feats[5] = tank_op.get('move_state', 0)
            # ... fill other features as needed
            
            # Label: 0-7 class
            # label_vec is [1, 0, 0, ...]
            label = np.argmax(label_vec)
            
            # IntentionLSTM expects [seq_len, features]. Let's fake seq_len=10 by repeating
            # Ideally we would link frames, but for this demo/mini-dataset structure, we simplify.
            seq_feats = np.tile(feats, (10, 1))
            
            self.samples.append({
                "features": seq_feats,
                "label": label
            })
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        return torch.tensor(item['features'], dtype=torch.float32), torch.tensor(item['label'], dtype=torch.long)

import ast

# ---------------------------------------------------------
# Situation Assessment Dataset (WG-StateGraph)
# ---------------------------------------------------------

class RealSituationDataset(Dataset):
    def __init__(self, root_dir):
        # Root dir: datasets/.../WG-StateGraph
        self.node_features_dir = os.path.join(root_dir, "encoded_nodes-features")
        
        # Labels
        label_file = os.path.join(root_dir, "label.csv")
        self.labels = pd.read_csv(label_file) # Assuming filename,label columns
        
        self.sample_files = []
        if os.path.exists(self.node_features_dir):
            for f in os.listdir(self.node_features_dir):
                if f.endswith(".csv"):
                    self.sample_files.append(os.path.join(self.node_features_dir, f))
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        file_path = self.sample_files[idx]
        basename = os.path.basename(file_path)
        # remove "encoded_" prefix and "_nodes_features.csv" suffix
        # Filename format: encoded_2021..._nodes_features.csv
        # Label filename: 2021... .json
        
        original_json_name = basename.replace("encoded_", "").replace("_nodes_features.csv", ".json")
        
        try:
            label_row = self.labels[self.labels['filename'] == original_json_name]
            if len(label_row) > 0:
                y = float(label_row.iloc[0, 1]) 
            else:
                y = 0.5
        except:
            y = 0.5
            
        # Read node features
        try:
            df = pd.read_csv(file_path)
            # Pick a random row (timestep)
            if len(df) > 0:
                row_idx = np.random.randint(0, len(df))
                row = df.iloc[row_idx]
                
                # Parse cells
                # Each cell is a stringified list "[...]"
                # We need to transpose this: Column = Feature, List idx = Node
                # Matrix shape: [Num_Nodes, Num_Features]
                
                parsed_cols = []
                for col in df.columns:
                    val = row[col]
                    try:
                        # Use ast.literal_eval for safety and correctness
                        parsed_list = ast.literal_eval(val)
                        # Ensure it's a list
                        if not isinstance(parsed_list, list):
                            parsed_list = [parsed_list] # Should not happen based on inspection
                    except:
                        # Fallback for non-list cells or errors
                        parsed_list = []
                    parsed_cols.append(parsed_list)
                
                # Assume all lists have same length (Num_Nodes)
                # Find max length just in case
                num_nodes = 0
                for lst in parsed_cols:
                    if len(lst) > num_nodes:
                        num_nodes = len(lst)
                
                # Build matrix
                num_feats = len(parsed_cols)
                x = np.zeros((num_nodes, num_feats), dtype=np.float32)
                
                for f_idx, lst in enumerate(parsed_cols):
                    # Fill features. If list is shorter, pad with 0? 
                    # Or maybe shorter list implies scalar feature?
                    # Inspection showed lists of same length.
                    length = len(lst)
                    if length == num_nodes:
                        # Handle non-numeric?
                        # Some lists had strings ['重型坦克', ...]. Need to handle or skip.
                        # GCN expects float.
                        # Simple strategy: Try convert to float, else 0.
                        # Or pre-filter columns.
                        # Let's try convert.
                        try:
                            # Check first element
                            float(lst[0])
                            x[:, f_idx] = lst
                        except:
                            # Non-numeric feature (name, etc.), skip or encode?
                            # For simplicity, skip/zero out.
                            pass
                    elif length == 1:
                        # Scalar broadcast?
                        try:
                            val = float(lst[0])
                            x[:, f_idx] = val
                        except:
                            pass
                
                # Pad/Cut to 108 features expected by model
                target_dim = 108
                if x.shape[1] < target_dim:
                    x = np.pad(x, ((0,0), (0, target_dim - x.shape[1])))
                elif x.shape[1] > target_dim:
                    x = x[:, :target_dim]
                    
            else:
                x = np.zeros((10, 108), dtype=np.float32)
                num_nodes = 10
        except Exception as e:
            # print(f"Error parsing {basename}: {e}")
            x = np.zeros((10, 108), dtype=np.float32)
            num_nodes = 10

        adj = np.eye(num_nodes, dtype=np.float32) 
        
        return torch.tensor(x), torch.tensor(adj), torch.tensor(y).unsqueeze(0)

def train_real():
    # 1. Train Intention
    try:
        print("Training IntentionLSTM with WargameData_mini01...")
        dataset = RealIntentionDataset("datasets/意图识别研究数据集WargameData_mini01/train.json")
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model = IntentionLSTM(input_dim=108, hidden_dim=256, num_classes=8) # 8 directions
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(3):
                total_loss = 0
                for x, y in loader:
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch}: Loss {total_loss/len(loader)}")
            
            torch.save(model.state_dict(), "ai/weights/intention_lstm.pth")
    except Exception as e:
        print(f"Intention training failed: {e}")

    # 2. Train Situation
    try:
        print("Training SituationGCN with WG-StateGraph...")
        dataset = RealSituationDataset("datasets/[重庆大学] 分队级陆战兵棋态势表征图数据集 WG-StateGraph")
        if len(dataset) > 0:
            # Batch size 1 for variable graph size
            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            model = SituationGCN(num_features=108, hidden_dim=64, output_dim=1)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss() # Binary Classification
            
            model.train()
            for epoch in range(3):
                total_loss = 0
                for x, adj, y in loader:
                    x, adj, y = x.squeeze(0), adj.squeeze(0), y.squeeze(0)
                    optimizer.zero_grad()
                    
                    # Forward
                    # SituationGCN returns [N, 1] node scores?
                    # Check models.py again. Yes, it returns sigmoid(fc(x)).
                    # We need graph-level score for Win/Loss prediction.
                    # Add pooling.
                    
                    node_scores = model(x, adj)
                    graph_score = torch.mean(node_scores, dim=0) # [1]
                    
                    loss = criterion(graph_score, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch}: Loss {total_loss/len(loader)}")
            
            torch.save(model.state_dict(), "ai/weights/situation_gcn.pth")
            
    except Exception as e:
        print(f"Situation training failed: {e}")

if __name__ == "__main__":
    train_real()
