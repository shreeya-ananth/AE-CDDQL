"""
Comprehensive evaluation pipeline comparing all cloud scheduling algorithms:
- Random Scheduling
- Round Robin (RR)
- Greedy Earliest Finish (Min-Min heuristic)
- Minimum Workload
- Tabular Q-Learning (QLS)
- Baseline CDDQL
- Proposed Full AE-CDDQL

Evaluates:
1. Multi-batch workload (5 batches x 9 tasks = 45 tasks, matching Figure 2 of paper)
2. Per-task execution times on the 9 benchmark tasks (matching Figure 1 of paper)
3. Epsilon sensitivity analysis (0.0 to 1.0, matching Figure 3 of paper)
Exports all evaluation data to CSV files.
"""
import os
import sys
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation.cloud_environment import CloudSimEnvironment
from simulation.workload_generator import (
    generate_paper_tasks,
    generate_multi_batch_workload,
    generate_synthetic_workload
)
from algorithms.classical_schedulers import (
    RandomScheduler,
    RoundRobinScheduler,
    GreedyEarliestFinishScheduler,
    MinWorkloadScheduler
)
from algorithms.q_learning import QLearningScheduler
from algorithms.cddql_baseline import BaselineCDDQLScheduler
from algorithms.ae_cddql import AECDQLScheduler


def train_qls_agent(episodes: int = 150, seed: int = 42) -> QLearningScheduler:
    """Trains tabular QLS agent on the cloud environment."""
    np.random.seed(seed)
    env = CloudSimEnvironment(num_vms=4, state_mode="baseline", reward_mode="baseline")
    agent = QLearningScheduler(num_vms=4, epsilon=0.5)
    for ep in range(1, episodes + 1):
        tasks = generate_paper_tasks(batch_id=0, shuffle=True, seed=seed + ep)
        state = env.reset(tasks)
        done = False
        while not done:
            action = agent.select_action(state, env=env)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
    return agent


def evaluate_scheduler_on_tasks(
    scheduler,
    tasks_generator_fn,
    state_mode: str = "baseline",
    is_rl: bool = False
) -> Dict[str, Any]:
    """Evaluates a scheduler on a generated workload."""
    env = CloudSimEnvironment(num_vms=4, state_mode=state_mode, reward_mode="multi_objective")
    tasks = tasks_generator_fn()
    state = env.reset(tasks)
    
    if is_rl and hasattr(scheduler, "epsilon"):
        orig_eps = scheduler.epsilon
        scheduler.epsilon = 0.0 # Greedy evaluation
    else:
        orig_eps = None
        
    done = False
    task_exec_records = []
    
    while not done:
        action = scheduler.select_action(state, env=env)
        next_state, _, done, info = env.step(action)
        task_exec_records.append(info)
        state = next_state
        
    if orig_eps is not None:
        scheduler.epsilon = orig_eps
        
    metrics = env.get_episode_metrics()
    metrics["per_task_info"] = task_exec_records
    return metrics


def run_comprehensive_evaluation(save_dir: str = "results", seed: int = 42):
    """
    Runs full benchmark evaluation across classical and RL schedulers.
    """
    os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
    models_dir = os.path.join(save_dir, "models")
    
    # Instantiate Schedulers
    schedulers = {
        "Random": (RandomScheduler(num_vms=4, seed=seed), "baseline", False),
        "Round Robin": (RoundRobinScheduler(num_vms=4), "baseline", False),
        "Greedy EFT": (GreedyEarliestFinishScheduler(num_vms=4), "baseline", False),
        "Min Workload": (MinWorkloadScheduler(num_vms=4), "baseline", False),
    }
    
    # Train QLS baseline
    print("--> Training Tabular QLS baseline...")
    qls_agent = train_qls_agent(episodes=150, seed=seed)
    schedulers["QLS"] = (qls_agent, "baseline", False)
    
    # Load or initialize Baseline CDDQL
    cddql_agent = BaselineCDDQLScheduler(state_dim=3, num_vms=4)
    cddql_weight_path = os.path.join(models_dir, "CDDQL-Baseline.pth")
    if os.path.exists(cddql_weight_path):
        cddql_agent.q1.load_state_dict(torch.load(cddql_weight_path, map_location="cpu"))
        print(f" Loaded weights for Baseline CDDQL from {cddql_weight_path}")
    schedulers["CDDQL (Paper)"] = (cddql_agent, "baseline", True)
    
    # Load or initialize Proposed AE-CDDQL
    ae_cddql_agent = AECDQLScheduler(state_dim=5, num_vms=4, use_adaptive_epsilon=True)
    ae_cddql_weight_path = os.path.join(models_dir, "AE-CDDQL.pth")
    if os.path.exists(ae_cddql_weight_path):
        ae_cddql_agent.q1.load_state_dict(torch.load(ae_cddql_weight_path, map_location="cpu"))
        print(f" Loaded weights for Proposed AE-CDDQL from {ae_cddql_weight_path}")
    schedulers["AE-CDDQL (Proposed)"] = (ae_cddql_agent, "energy_aware", True)
    
    # -------------------------------------------------------------
    # Experiment 1: Multi-Batch Benchmark (5 batches x 9 tasks = 45 tasks)
    # -------------------------------------------------------------
    print("\n--- Evaluating Multi-Batch Benchmark (5 Batches of 9 Tasks = 45 Tasks) ---")
    multi_batch_results = []
    
    for name, (sched, state_mode, is_rl) in schedulers.items():
        sched.reset()
        gen_fn = lambda: generate_multi_batch_workload(num_batches=5, shuffle=False, seed=seed)
        m = evaluate_scheduler_on_tasks(sched, gen_fn, state_mode=state_mode, is_rl=is_rl)
        
        multi_batch_results.append({
            "scheduler": name,
            "makespan_sec": round(m["makespan"], 2),
            "avg_execution_time_sec": round(m["avg_execution_time"], 2),
            "total_execution_time_sec": round(m["total_execution_time"], 2),
            "avg_waiting_time_sec": round(m["avg_waiting_time"], 2),
            "total_waiting_time_sec": round(m["total_waiting_time"], 2),
            "total_energy_joules": round(m["total_energy_joules"], 2),
            "total_energy_kwh": round(m["total_energy_kwh"], 5),
            "total_cost_usd": round(m["total_cost"], 5),
            "avg_vm_utilization_pct": round(m["avg_vm_utilization"] * 100.0, 2)
        })
        
    df_multi = pd.DataFrame(multi_batch_results)
    csv_multi = os.path.join(save_dir, "csv", "multi_batch_evaluation.csv")
    df_multi.to_csv(csv_multi, index=False)
    print(df_multi.to_string(index=False))
    print(f" Saved multi-batch evaluation results to {csv_multi}")
    
    # -------------------------------------------------------------
    # Experiment 2: Per-Task Execution Time on Single Batch of 9 Tasks (Fig 1)
    # -------------------------------------------------------------
    print("\n--- Evaluating Per-Task Execution Times for Figure 1 Replication ---")
    per_task_records = []
    for name, (sched, state_mode, is_rl) in schedulers.items():
        sched.reset()
        gen_fn = lambda: generate_paper_tasks(batch_id=0, shuffle=False, seed=123)
        m = evaluate_scheduler_on_tasks(sched, gen_fn, state_mode=state_mode, is_rl=is_rl)
        for task_info in m["per_task_info"]:
            per_task_records.append({
                "scheduler": name,
                "task_id": task_info["task_id"],
                "vm_id": task_info["vm_id"],
                "exec_time_sec": round(task_info["exec_time"], 2),
                "wait_time_sec": round(task_info["wait_time"], 2),
                "energy_joules": round(task_info["energy"], 2)
            })
            
    df_per_task = pd.DataFrame(per_task_records)
    csv_per_task = os.path.join(save_dir, "csv", "per_task_execution_times.csv")
    df_per_task.to_csv(csv_per_task, index=False)
    print(f" Saved per-task execution data to {csv_per_task}")
    
    # -------------------------------------------------------------
    # Experiment 3: Epsilon Sensitivity Sweep (Fig 3 Replication)
    # -------------------------------------------------------------
    print("\n--- Evaluating Epsilon Sensitivity Sweep (0.0 to 1.0) ---")
    eps_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    eps_records = []
    for eps in eps_values:
        # Run 10 trials per epsilon for statistical stability
        wait_times = []
        makespans = []
        for trial in range(10):
            env = CloudSimEnvironment(num_vms=4, state_mode="baseline", reward_mode="baseline")
            tasks = generate_paper_tasks(batch_id=0, shuffle=True, seed=trial * 100)
            state = env.reset(tasks)
            done = False
            while not done:
                # With probability eps, choose random; else best known
                if np.random.rand() < eps:
                    action = np.random.randint(4)
                else:
                    state_t = torch.from_numpy(state).float().unsqueeze(0)
                    with torch.no_grad():
                        action = int(torch.argmax(cddql_agent.q1(state_t), dim=1).item())
                state, _, done, _ = env.step(action)
            m = env.get_episode_metrics()
            wait_times.append(m["avg_waiting_time"])
            makespans.append(m["makespan"])
            
        eps_records.append({
            "epsilon": eps,
            "avg_waiting_time_sec": round(float(np.mean(wait_times)), 2),
            "std_waiting_time_sec": round(float(np.std(wait_times)), 2),
            "avg_makespan_sec": round(float(np.mean(makespans)), 2)
        })
        
    df_eps = pd.DataFrame(eps_records)
    csv_eps = os.path.join(save_dir, "csv", "epsilon_sensitivity.csv")
    df_eps.to_csv(csv_eps, index=False)
    print(df_eps.to_string(index=False))
    print(f" Saved epsilon sensitivity sweep results to {csv_eps}")
    
    return df_multi, df_per_task, df_eps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cloud Task Schedulers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_comprehensive_evaluation(seed=args.seed)
