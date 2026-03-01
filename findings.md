# Findings

## Real Dataset Training Results
- **IntentionLSTM**:
    - Trained on `WargameData_mini01` (Real Data).
    - Loss: 2.05 -> 1.21 (3 Epochs).
    - Saved to `ai/weights/intention_lstm.pth`.
- **SituationGCN**:
    - Still silently failing/exiting loop.
    - `len(dataset)` must be 0.
    - `sample_files = glob.glob(os.path.join(self.node_features_dir, "*.csv"))`.
    - `node_features_dir` = `datasets/[重庆大学] 分队级陆战兵棋态势表征图数据集 WG-StateGraph/encoded_nodes-features`.
    - The LS output confirmed the files exist.
    - Potential issue: The square brackets `[]` in the path are wildcards in `glob`.
    - `[重庆大学]` might be interpreted as "match one character from '重庆大学'".
    - **Fix**: Escape the brackets in `glob` pattern or don't use glob on the folder part.
    - Better: `glob.glob(os.path.join(glob.escape(self.node_features_dir), "*.csv"))`
    - Or just `os.listdir` and filter.

## Action Plan Update
1.  **Fix GCN Loader Path**:
    - Use `os.listdir` instead of `glob` for safety with special chars in directory names.
2.  **Retrain**.
3.  **Run Evaluation**.
