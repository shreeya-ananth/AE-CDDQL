"""
Training pipeline for Baseline CDDQL, Proposed AE-CDDQL, and Ablation Models.
Implements Algorithm 1 & Algorithm 2 from the baseline paper with full support for:
- Fixed vs Adaptive Epsilon
- Baseline (3D) vs Energy-Aware (5D) State
- Discrete Rank-based vs Multi-Objective Reward
"""
import os
import sys
import argparse
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation.cloud_environment import CloudSimEnvironment
from simulation.workload_generator import generate_paper_tasks
from algorithms.cddql_baseline import BaselineCDDQLScheduler
from algorithms.ae_cddql import AECDQLScheduler


def train_agent(
    model_name: str,
    num_episodes: int = 200,
    state_mode: str = "baseline",
    reward_mode: str = "baseline",
    use_adaptive_epsilon: bool = False,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_dir: str = "results",
    seed: int = 42
) -> pd.DataFrame:
    """
    Train a single DRL scheduling agent on cloud task episodes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    
    env = CloudSimEnvironment(
        num_vms=4,
        state_mode=state_mode,
        reward_mode=reward_mode
    )
    
    state_dim = env.state_dim
    
    if model_name == "CDDQL-Baseline":
        agent = BaselineCDDQLScheduler(
            state_dim=state_dim,
            num_vms=4,
            lr=lr,
            batch_size=batch_size
        )
    else:
        agent = AECDQLScheduler(
            state_dim=state_dim,
            num_vms=4,
            lr=lr,
            batch_size=batch_size,
            use_adaptive_epsilon=use_adaptive_epsilon
        )
        
    records: List[Dict[str, Any]] = []
    print(f"--> Starting Training: {model_name} [Episodes: {num_episodes}, StateDim: {state_dim}, Reward: {reward_mode}, AdaptiveEps: {use_adaptive_epsilon}]")
    start_time = time.time()
    
    for ep in range(1, num_episodes + 1):
        # Algorithm 2 line 3.2: Shuffle primitive task set
        tasks = generate_paper_tasks(batch_id=0, shuffle=True, seed=seed + ep)
        state = env.reset(tasks)
        
        cumulative_reward = 0.0
        losses = []
        done = False
        
        while not done:
            action = agent.select_action(state, env=env)
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward
            
            # Store in experience replay memory
            agent.store_transition(state, action, reward, next_state, done)
            
            # Perform gradient descent update
            update_info = agent.update()
            if update_info and "loss1" in update_info:
                losses.append(update_info["loss1"])
                
            state = next_state
            
        metrics = env.get_episode_metrics()
        avg_loss = float(np.mean(losses)) if losses else 0.0
        current_eps = float(agent.epsilon)
        
        records.append({
            "episode": ep,
            "model": model_name,
            "cumulative_reward": cumulative_reward,
            "avg_loss": avg_loss,
            "epsilon": current_eps,
            "makespan": metrics["makespan"],
            "total_energy_joules": metrics["total_energy_joules"],
            "avg_waiting_time": metrics["avg_waiting_time"],
            "total_cost": metrics["total_cost"],
            "avg_vm_utilization": metrics["avg_vm_utilization"]
        })
        
        if ep % 25 == 0 or ep == num_episodes:
            print(f"   Ep {ep:3d}/{num_episodes:3d} | Reward: {cumulative_reward:6.2f} | Loss: {avg_loss:6.4f} | Eps: {current_eps:.3f} | Makespan: {metrics['makespan']:.2f}s | Energy: {metrics['total_energy_joules']:.1f}J")
            
    elapsed = time.time() - start_time
    print(f" Completed {model_name} training in {elapsed:.1f}s.")
    
    # Save model weights
    model_path = os.path.join(save_dir, "models", f"{model_name}.pth")
    torch.save(agent.q1.state_dict(), model_path)
    
    # Save training logs to CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, "csv", f"training_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f" Saved training history to {csv_path}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cloud Scheduling DRL Agents")
    parser.add_argument("--model", type=str, default="AE-CDDQL", choices=["CDDQL-Baseline", "AE-CDDQL", "all"])
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.model == "all":
        # Train baseline
        train_agent("CDDQL-Baseline", num_episodes=args.episodes, state_mode="baseline", reward_mode="baseline", use_adaptive_epsilon=False, seed=args.seed)
        # Train proposed AE-CDDQL
        train_agent("AE-CDDQL", num_episodes=args.episodes, state_mode="energy_aware", reward_mode="multi_objective", use_adaptive_epsilon=True, seed=args.seed)
    elif args.model == "CDDQL-Baseline":
        train_agent("CDDQL-Baseline", num_episodes=args.episodes, state_mode="baseline", reward_mode="baseline", use_adaptive_epsilon=False, seed=args.seed)
    else:
        train_agent("AE-CDDQL", num_episodes=args.episodes, state_mode="energy_aware", reward_mode="multi_objective", use_adaptive_epsilon=True, seed=args.seed)
