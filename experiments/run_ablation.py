"""
Dedicated Ablation Study Pipeline for AE-CDDQL.
Directly implements the 5 variants required by the user:
1. CDDQL (Baseline)
2. CDDQL + Adaptive ε
3. CDDQL + Energy State
4. CDDQL + Multi-Objective Reward
5. Full AE-CDDQL (All 3 modifications combined)
"""
import os
import sys
import argparse
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation.cloud_environment import CloudSimEnvironment
from simulation.workload_generator import generate_paper_tasks, generate_multi_batch_workload
from algorithms.ae_cddql import AECDQLScheduler


ABLATION_CONFIGS = [
    {
        "name": "CDDQL",
        "description": "Baseline CDDQL (Fixed eps=0.5, 3D State, Baseline Reward)",
        "state_mode": "baseline",
        "state_dim": 3,
        "use_adaptive_eps": False,
        "reward_mode": "baseline"
    },
    {
        "name": "CDDQL + Adaptive eps",
        "description": "Baseline + Adaptive Epsilon decay",
        "state_mode": "baseline",
        "state_dim": 3,
        "use_adaptive_eps": True,
        "reward_mode": "baseline"
    },
    {
        "name": "CDDQL + Energy State",
        "description": "Baseline + 5D Energy-Aware State",
        "state_mode": "energy_aware",
        "state_dim": 5,
        "use_adaptive_eps": False,
        "reward_mode": "baseline"
    },
    {
        "name": "CDDQL + MO Reward",
        "description": "Baseline + Multi-Objective Reward",
        "state_mode": "baseline",
        "state_dim": 3,
        "use_adaptive_eps": False,
        "reward_mode": "multi_objective"
    },
    {
        "name": "Full AE-CDDQL",
        "description": "Proposed Full AE-CDDQL (All 3 Modifications)",
        "state_mode": "energy_aware",
        "state_dim": 5,
        "use_adaptive_eps": True,
        "reward_mode": "multi_objective"
    }
]


def run_ablation_study(
    episodes: int = 150,
    eval_batches: int = 5,
    save_dir: str = "results",
    seed: int = 42
) -> pd.DataFrame:
    """
    Trains and evaluates each ablation configuration under identical conditions.
    """
    os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    
    results = []
    
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        print(f"\n==================================================")
        print(f"Running Ablation Variant: {name}")
        print(f"Details: {cfg['description']}")
        print(f"==================================================")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = CloudSimEnvironment(
            num_vms=4,
            state_mode=cfg["state_mode"],
            reward_mode=cfg["reward_mode"]
        )
        
        agent = AECDQLScheduler(
            state_dim=cfg["state_dim"],
            num_vms=4,
            use_adaptive_epsilon=cfg["use_adaptive_eps"],
            fixed_epsilon=0.5
        )
        agent.name = name
        
        # 1. Training Phase
        t0 = time.time()
        for ep in range(1, episodes + 1):
            tasks = generate_paper_tasks(batch_id=0, shuffle=True, seed=seed + ep)
            state = env.reset(tasks)
            done = False
            while not done:
                action = agent.select_action(state, env=env)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                
        train_duration = time.time() - t0
        print(f"Finished training in {train_duration:.2f}s.")
        
        # Save model
        clean_filename = name.replace(" ", "_").replace("+", "plus")
        model_path = os.path.join(save_dir, "models", f"ablation_{clean_filename}.pth")
        torch.save(agent.q1.state_dict(), model_path)
        
        # 2. Evaluation Phase (Greedy exploitation, eps=0)
        # Evaluate on the multi-batch benchmark matching Figure 2 of paper (5 batches of 9 tasks = 45 tasks)
        # Standardized evaluation environment
        eval_env = CloudSimEnvironment(
            num_vms=4,
            state_mode=cfg["state_mode"],
            reward_mode="multi_objective" # standard metric evaluation
        )
        
        eval_tasks = generate_multi_batch_workload(num_batches=eval_batches, shuffle=False, seed=999)
        state = eval_env.reset(eval_tasks)
        
        # Evaluation with exploitation
        original_eps = agent.epsilon
        agent.epsilon = 0.0 # pure exploitation
        
        done = False
        while not done:
            action = agent.select_action(state, env=eval_env)
            next_state, _, done, _ = eval_env.step(action)
            state = next_state
            
        agent.epsilon = original_eps
        metrics = eval_env.get_episode_metrics()
        
        res = {
            "variant": name,
            "description": cfg["description"],
            "state_mode": cfg["state_mode"],
            "adaptive_eps": cfg["use_adaptive_eps"],
            "reward_mode": cfg["reward_mode"],
            "makespan_sec": round(metrics["makespan"], 2),
            "avg_execution_time_sec": round(metrics["avg_execution_time"], 2),
            "total_execution_time_sec": round(metrics["total_execution_time"], 2),
            "avg_waiting_time_sec": round(metrics["avg_waiting_time"], 2),
            "total_waiting_time_sec": round(metrics["total_waiting_time"], 2),
            "total_energy_joules": round(metrics["total_energy_joules"], 2),
            "total_energy_kwh": round(metrics["total_energy_kwh"], 5),
            "total_cost_usd": round(metrics["total_cost"], 5),
            "avg_vm_utilization_pct": round(metrics["avg_vm_utilization"] * 100.0, 2),
        }
        results.append(res)
        print(f"--> Result: Makespan={res['makespan_sec']}s | Wait={res['avg_waiting_time_sec']}s | Energy={res['total_energy_joules']}J | Cost=${res['total_cost_usd']} | Util={res['avg_vm_utilization_pct']}%")
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "csv", "ablation_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n Ablation study successfully completed! Saved results to {csv_path}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Study for AE-CDDQL")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_ablation_study(episodes=args.episodes, eval_batches=args.batches, seed=args.seed)
