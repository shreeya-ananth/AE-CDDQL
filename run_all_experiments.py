"""
Master execution script for the entire project.
Runs:
1. Agent Training (Baseline CDDQL and Proposed AE-CDDQL)
2. Complete 5-variant Ablation Study
3. Comprehensive Evaluation across all Classical and RL Schedulers
4. Generation of all Publication-Quality Plots and CSV Summary Tables
"""
import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from experiments.train import train_agent
from experiments.run_ablation import run_ablation_study
from experiments.evaluate import run_comprehensive_evaluation
from experiments.plot_results import generate_all_plots


def main():
    print("================================================================================")
    print("  Adaptive Energy-Aware Clipped Double Deep Q-Learning (AE-CDDQL) Project")
    print("  Cloud Task Scheduling Framework & Experimental Evaluation Pipeline")
    print("================================================================================\n")
    
    total_start = time.time()
    save_dir = "results"
    os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    
    episodes = 150
    seed = 42
    
    # ---------------------------------------------------------
    # Stage 1: Train Baseline CDDQL & Proposed AE-CDDQL
    # ---------------------------------------------------------
    print("\n>>> STAGE 1: Training Primary DRL Agents <<<")
    print("1. Training Baseline CDDQL (Fixed eps=0.5, 3D State, Baseline Reward)...")
    train_agent(
        model_name="CDDQL-Baseline",
        num_episodes=episodes,
        state_mode="baseline",
        reward_mode="baseline",
        use_adaptive_epsilon=False,
        save_dir=save_dir,
        seed=seed
    )
    
    print("\n2. Training Proposed AE-CDDQL (Adaptive eps, 5D Energy State, Multi-Objective Reward)...")
    train_agent(
        model_name="AE-CDDQL",
        num_episodes=episodes,
        state_mode="energy_aware",
        reward_mode="multi_objective",
        use_adaptive_epsilon=True,
        save_dir=save_dir,
        seed=seed
    )
    
    # ---------------------------------------------------------
    # Stage 2: Ablation Study
    # ---------------------------------------------------------
    print("\n>>> STAGE 2: Executing Ablation Study (5 Configurations) <<<")
    run_ablation_study(episodes=episodes, eval_batches=5, save_dir=save_dir, seed=seed)
    
    # ---------------------------------------------------------
    # Stage 3: Comprehensive Multi-Scheduler Evaluation
    # ---------------------------------------------------------
    print("\n>>> STAGE 3: Comprehensive Scheduler Evaluation <<<")
    run_comprehensive_evaluation(save_dir=save_dir, seed=seed)
    
    # ---------------------------------------------------------
    # Stage 4: Plot Generation
    # ---------------------------------------------------------
    print("\n>>> STAGE 4: Generating Publication-Quality Figures <<<")
    generate_all_plots(save_dir=save_dir)
    
    total_duration = time.time() - total_start
    print("\n================================================================================")
    print(f" ALL EXPERIMENTS & PLOTS COMPLETED IN {total_duration:.1f} SECONDS!")
    print(f" Results available in: {os.path.abspath(save_dir)}")
    print("================================================================================")


if __name__ == "__main__":
    main()
