# AI Model Evaluation Report
**Date:** 2026-03-01
**Environment:** Linux Docker Container (Standardized)
**Test Subject:** CurrentAI (Trained Models) vs DemoAI (Baseline)

## 1. Executive Summary
The AI models (`IntentionLSTM` and `SituationGCN`) were successfully trained on synthetic datasets and evaluated in a standardized Linux container environment. The evaluation process confirmed that the models are operational and can be loaded by the agent. However, due to the symmetric nature of the test scenario and the simplified behavior of the synthetic data, the tactical outcome remains a draw, similar to the heuristic baseline.

## 2. Test Configuration
- **Total Rounds:** 20
- **Max Steps per Round:** 100
- **Agents:**
    - **Red (CurrentAI):** Enabled with trained `IntentionLSTM` and `SituationGCN` models.
    - **Blue (DemoAI):** Standard Rule-based logic.
- **Hardware:** Local Windows Workstation (Dockerized Linux x86_64)

## 3. Performance Metrics

### 3.1 Tactical Performance
| Metric | Red (Optimized) | Blue (Baseline) | Delta |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 0% | 0% | 0% (100% Draw) |
| **Avg Score** | 1000.0 | 1000.0 | 0.0 |
| **Action Validity** | 100% | 100% | - |

> **Analysis:** The trained models were loaded successfully, but in the simplified 1v1 mock environment, the optimal strategy (Shoot when possible) is trivial. Both agents executed this strategy perfectly, resulting in a draw. The "tactical intelligence emergence" requires more complex scenarios (terrain, fog of war, multiple units) which are not fully represented in the mock environment or the synthetic training data.

### 3.2 Computational Efficiency
- **Total Duration (20 rounds):** ~4.7 seconds
- **Avg Latency per Step:** **2.35 ms** (Increased from 0.27ms due to Neural Network inference)
- **Throughput:** ~425 steps/second

> **Analysis:** The inference latency increased by ~2ms per step compared to pure rule-based logic. This is expected and well within the 50ms real-time requirement. It confirms that the Python-based inference (even without TensorRT optimization in this specific run) is efficient enough.

## 4. Verification Conclusion
- **Environment:** Successfully deployed a Linux-compatible container capable of running the `land_wargame_sdk`.
- **Training:** Successfully implemented a training pipeline and generated model weights.
- **Integration:** Successfully integrated trained models into the Agent's decision loop.

## 5. Next Steps
- **Data:** Acquire real gameplay data (`WG-StateGraph`, `WargameData_mini01`) to train models on meaningful patterns.
- **Scenario:** Test on complex maps (e.g., city combat) where `SituationGCN` can demonstrate its value in path planning and threat assessment.
