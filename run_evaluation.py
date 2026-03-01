import json
import pickle
import time
import os
import sys
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path to import ai module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try imports
try:
    from ai.agent import Agent
    import land_wargame_train_env
    from land_wargame_train_env import TrainEnv
    logging.info("Successfully imported Linux TrainEnv")
except ImportError:
    try:
        # Fallback for different directory structures or Windows dev environment
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'land_wargame_sdk'))
        from ai.agent import Agent
        from train_env import TrainEnv
        logging.info("Imported local SDK TrainEnv")
    except ImportError:
        logging.warning("Standard TrainEnv not found (Windows/Linux mismatch?). Using MockTrainEnv for logic verification.")
        try:
            from mock_train_env import MockTrainEnv as TrainEnv
            from ai.agent import Agent
            logging.info("Using MockTrainEnv")
        except ImportError:
            logging.error("Failed to import MockTrainEnv. Cannot run evaluation.")
            sys.exit(1)


RED, BLUE = 0, 1

def setup_logging(log_dir="logs/evaluation"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eval_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_evaluation(num_rounds=20):
    log_file = setup_logging()
    logging.info(f"Starting Evaluation: {num_rounds} rounds")
    logging.info("Red: CurrentAI (Optimized) vs Blue: DemoAI (Baseline)")
    
    # Load Data
    data_dir = "land_wargame_sdk/Data/Data"
    if not os.path.exists(data_dir):
        data_dir = "Data/Data"

    try:
        with open(f"{data_dir}/scenarios/1231.json", encoding='utf8') as f:
            scenario_data = json.load(f)
        with open(f"{data_dir}/maps/map_123/basic.json", encoding='utf8') as f:
            basic_data = json.load(f)
        with open(f"{data_dir}/maps/map_123/cost.pickle", 'rb') as file:
            cost_data = pickle.load(file)
        see_data = np.load(f"{data_dir}/maps/map_123/123see.npz")['data']
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    env = TrainEnv()
    player_info = [
        {"seat": 1, "faction": 0, "role": 1, "user_name": "CurrentAI", "user_id": 1},
        {"seat": 11, "faction": 1, "role": 1, "user_name": "DemoAI", "user_id": 11}
    ]
    env_step_info = {
        "scenario_data": scenario_data, "basic_data": basic_data,
        "cost_data": cost_data, "see_data": see_data, "player_info": player_info
    }

    results = {
        "red_wins": 0, "blue_wins": 0, "draws": 0,
        "red_scores": [], "blue_scores": [],
        "steps": [], "durations": []
    }

    for i in range(num_rounds):
        logging.info(f"--- Round {i+1}/{num_rounds} ---")
        start_time = time.time()
        
        state = env.setup(env_step_info)
        red = Agent()
        blue = Agent()
        
        # Setup Agents
        # Red gets "CurrentAI" name to trigger optimization
        red.setup({"seat": 1, "faction": 0, "role": 1, "user_name": "CurrentAI", "state": state, 
                   "scenario": scenario_data, "basic_data": basic_data, "cost_data": cost_data, "see_data": see_data})
        # Blue gets "DemoAI" name (defaults to standard rules)
        blue.setup({"seat": 11, "faction": 1, "role": 1, "user_name": "DemoAI", "state": state,
                    "scenario": scenario_data, "basic_data": basic_data, "cost_data": cost_data, "see_data": see_data})
        
        done = False
        step_count = 0
        while not done and step_count < 1000:
            actions = red.step(state[RED]) + blue.step(state[BLUE])
            state, done = env.step(actions)
            step_count += 1
        
        duration = time.time() - start_time
        r_score = state[RED].get('reward', 0)
        b_score = state[BLUE].get('reward', 0)
        
        results["red_scores"].append(r_score)
        results["blue_scores"].append(b_score)
        results["steps"].append(step_count)
        results["durations"].append(duration)
        
        outcome = "Draw"
        if r_score > b_score:
            results["red_wins"] += 1
            outcome = "Red Win"
        elif b_score > r_score:
            results["blue_wins"] += 1
            outcome = "Blue Win"
        else:
            results["draws"] += 1
            
        logging.info(f"Result: {outcome} | Score: {r_score:.1f} - {b_score:.1f} | Steps: {step_count} | Time: {duration:.2f}s")

    # Analysis
    avg_red = np.mean(results["red_scores"])
    avg_blue = np.mean(results["blue_scores"])
    win_rate = results["red_wins"] / num_rounds
    
    summary = f"""
    === Evaluation Summary ===
    Total Rounds: {num_rounds}
    Red (Optimized) Wins: {results['red_wins']} ({win_rate:.1%})
    Blue (Baseline) Wins: {results['blue_wins']} ({results['blue_wins']/num_rounds:.1%})
    Draws: {results['draws']}
    Avg Score: Red {avg_red:.1f} vs Blue {avg_blue:.1f}
    Avg Steps: {np.mean(results['steps']):.1f}
    Avg Duration: {np.mean(results['durations']):.2f}s
    """
    logging.info(summary)
    
    # Save detailed results
    res_file = os.path.join(os.path.dirname(log_file), "results.json")
    with open(res_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Detailed results saved to {res_file}")

if __name__ == "__main__":
    run_evaluation()
