# Data Inspection
import os
import json
import numpy as np

def inspect_datasets():
    base_dir = "datasets"
    state_graph_dir = os.path.join(base_dir, "WG-StateGraph")
    mini_data_dir = os.path.join(base_dir, "WargameData_mini01")
    
    print(f"Inspecting {state_graph_dir}...")
    if os.path.exists(state_graph_dir):
        files = os.listdir(state_graph_dir)[:5]
        print(f"Files: {files}")
        # Try to read one
        if files:
            file_path = os.path.join(state_graph_dir, files[0])
            try:
                if file_path.endswith(".json"):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        print(f"Sample JSON keys: {list(data.keys()) if isinstance(data, dict) else 'List'}")
                elif file_path.endswith(".npy") or file_path.endswith(".npz"):
                     data = np.load(file_path, allow_pickle=True)
                     print(f"Sample Numpy shape: {data.shape}")
            except Exception as e:
                print(f"Error reading file: {e}")
    else:
        print("Directory not found.")

    print(f"\nInspecting {mini_data_dir}...")
    if os.path.exists(mini_data_dir):
        files = os.listdir(mini_data_dir)[:5]
        print(f"Files: {files}")
    else:
        print("Directory not found.")

if __name__ == "__main__":
    inspect_datasets()
