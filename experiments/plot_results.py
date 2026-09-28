"""
Plotting module generating publication-quality figures:
1. Training curves (Reward, Loss, Epsilon, Makespan)
2. Multi-scheduler performance comparison (Makespan, Energy, Waiting Time, Cost)
3. Ablation study breakdown (5 variants)
4. Replication of Paper Figure 1 (Per-task execution time)
5. Replication of Paper Figure 2 (Batch execution time)
6. Replication of Paper Figure 3 (Epsilon sensitivity analysis)
7. Multi-objective trade-off radar chart
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#e6ab02", "#1b9e77", "#e41a1c"]


def plot_training_curves(save_dir: str = "results"):
    csv_cddql = os.path.join(save_dir, "csv", "training_CDDQL-Baseline.csv")
    csv_aecddql = os.path.join(save_dir, "csv", "training_AE-CDDQL.csv")
    
    if not os.path.exists(csv_cddql) or not os.path.exists(csv_aecddql):
        print("Training CSV files not found. Skipping training curve plots.")
        return
        
    df_cddql = pd.read_csv(csv_cddql)
    df_aecddql = pd.read_csv(csv_aecddql)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # 1. Cumulative Reward (Smoothed)
    window = 10
    axes[0, 0].plot(df_cddql["episode"], df_cddql["cumulative_reward"].rolling(window, min_periods=1).mean(),
                    label="CDDQL (Baseline)", color="#e7298a", lw=2)
    axes[0, 0].plot(df_aecddql["episode"], df_aecddql["cumulative_reward"].rolling(window, min_periods=1).mean(),
                    label="AE-CDDQL (Proposed)", color="#1b9e77", lw=2)
    axes[0, 0].set_title("(a) Cumulative Reward per Episode (Rolling Mean)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    # 2. TD Error / Loss
    axes[0, 1].plot(df_cddql["episode"], df_cddql["avg_loss"].rolling(window, min_periods=1).mean(),
                    label="CDDQL Loss", color="#e7298a", lw=2)
    axes[0, 1].plot(df_aecddql["episode"], df_aecddql["avg_loss"].rolling(window, min_periods=1).mean(),
                    label="AE-CDDQL Loss", color="#1b9e77", lw=2)
    axes[0, 1].set_title("(b) Mean Squared TD Error Loss")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("MSE Loss")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    # 3. Epsilon Exploration Schedule
    axes[1, 0].plot(df_cddql["episode"], df_cddql["epsilon"],
                    label="Baseline: Fixed ε = 0.5", color="#e7298a", lw=2, linestyle="--")
    axes[1, 0].plot(df_aecddql["episode"], df_aecddql["epsilon"],
                    label="Proposed: Adaptive ε Decay", color="#1b9e77", lw=2)
    axes[1, 0].set_title("(c) Exploration-Exploitation Strategy (ε Parameter)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon Value")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    # 4. Energy Consumption during Training
    axes[1, 1].plot(df_cddql["episode"], df_cddql["total_energy_joules"].rolling(window, min_periods=1).mean(),
                    label="CDDQL Energy", color="#e7298a", lw=2)
    axes[1, 1].plot(df_aecddql["episode"], df_aecddql["total_energy_joules"].rolling(window, min_periods=1).mean(),
                    label="AE-CDDQL Energy", color="#1b9e77", lw=2)
    axes[1, 1].set_title("(d) Total Energy Consumption (Joules)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "plots", "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Saved training curves to {plot_path}")


def plot_multi_batch_comparison(save_dir: str = "results"):
    csv_path = os.path.join(save_dir, "csv", "multi_batch_evaluation.csv")
    if not os.path.exists(csv_path):
        print("multi_batch_evaluation.csv not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    schedulers = df["scheduler"].tolist()
    colors = sns.color_palette("deep", len(schedulers))
    
    # (a) Makespan
    axes[0, 0].bar(schedulers, df["makespan_sec"], color=colors, edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("(a) Makespan (Total Completion Time)", fontweight="bold")
    axes[0, 0].set_ylabel("Time (seconds)")
    axes[0, 0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(df["makespan_sec"]):
        axes[0, 0].text(i, v + 1.0, f"{v:.1f}s", ha="center", fontsize=9, fontweight="bold")
        
    # (b) Average Waiting Time
    axes[0, 1].bar(schedulers, df["avg_waiting_time_sec"], color=colors, edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("(b) Average Task Waiting Time", fontweight="bold")
    axes[0, 1].set_ylabel("Waiting Time (seconds)")
    axes[0, 1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(df["avg_waiting_time_sec"]):
        axes[0, 1].text(i, v + 0.3, f"{v:.2f}s", ha="center", fontsize=9, fontweight="bold")
        
    # (c) Total Energy Consumption (Joules)
    axes[1, 0].bar(schedulers, df["total_energy_joules"], color=colors, edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("(c) Datacenter Energy Consumption", fontweight="bold")
    axes[1, 0].set_ylabel("Total Energy (Joules)")
    axes[1, 0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(df["total_energy_joules"]):
        axes[1, 0].text(i, v + 50.0, f"{int(v)}J", ha="center", fontsize=9, fontweight="bold")
        
    # (d) Average VM Utilization
    axes[1, 1].bar(schedulers, df["avg_vm_utilization_pct"], color=colors, edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("(d) Average VM Resource Utilization", fontweight="bold")
    axes[1, 1].set_ylabel("Utilization (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(df["avg_vm_utilization_pct"]):
        axes[1, 1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "plots", "multi_scheduler_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Saved multi-scheduler comparison to {plot_path}")


def plot_ablation_study(save_dir: str = "results"):
    csv_path = os.path.join(save_dir, "csv", "ablation_study.csv")
    if not os.path.exists(csv_path):
        print("ablation_study.csv not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    variants = df["variant"].tolist()
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # 1. Makespan
    axes[0].barh(variants, df["makespan_sec"], color=colors, edgecolor="black")
    axes[0].set_title("Makespan (Lower is Better)", fontweight="bold")
    axes[0].set_xlabel("Seconds")
    for i, v in enumerate(df["makespan_sec"]):
        axes[0].text(v + 1, i, f"{v:.1f}s", va="center", fontweight="bold")
        
    # 2. Total Energy
    axes[1].barh(variants, df["total_energy_joules"], color=colors, edgecolor="black")
    axes[1].set_title("Total Energy (Lower is Better)", fontweight="bold")
    axes[1].set_xlabel("Joules")
    for i, v in enumerate(df["total_energy_joules"]):
        axes[1].text(v + 50, i, f"{int(v)}J", va="center", fontweight="bold")
        
    # 3. Average Waiting Time
    axes[2].barh(variants, df["avg_waiting_time_sec"], color=colors, edgecolor="black")
    axes[2].set_title("Avg Waiting Time (Lower is Better)", fontweight="bold")
    axes[2].set_xlabel("Seconds")
    for i, v in enumerate(df["avg_waiting_time_sec"]):
        axes[2].text(v + 0.2, i, f"{v:.2f}s", va="center", fontweight="bold")
        
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "plots", "ablation_study_breakdown.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Saved ablation study plot to {plot_path}")


def plot_paper_figures_replication(save_dir: str = "results"):
    """
    Replicates Figures 1 and 3 from the baseline paper:
    - Fig 1: Execution time of each user task (tasks 1 to 9).
    - Fig 3: Waiting time or delay of a batch of 9 tasks with respect to changing epsilon values.
    """
    # Replication of Figure 1
    csv_per_task = os.path.join(save_dir, "csv", "per_task_execution_times.csv")
    if os.path.exists(csv_per_task):
        df_task = pd.read_csv(csv_per_task)
        
        plt.figure(figsize=(13, 6))
        schedulers = ["Random", "QLS", "CDDQL (Paper)", "AE-CDDQL (Proposed)"]
        df_filtered = df_task[df_task["scheduler"].isin(schedulers)]
        
        palette = {"Random": "#7f7f7f", "QLS": "#e6ab02", "CDDQL (Paper)": "#386cb0", "AE-CDDQL (Proposed)": "#1b9e77"}
        sns.barplot(data=df_filtered, x="task_id", y="exec_time_sec", hue="scheduler", palette=palette, edgecolor="black")
        plt.title("Replication of Paper Fig. 1: Execution Time of Each User Task (1 to 9)", fontweight="bold")
        plt.xlabel("Task ID (Google Task Events Benchmark)")
        plt.ylabel("Execution Time (seconds)")
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "plots", "paper_fig1_per_task_execution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f" Saved Paper Figure 1 replication to {plot_path}")

    # Replication of Figure 3
    csv_eps = os.path.join(save_dir, "csv", "epsilon_sensitivity.csv")
    if os.path.exists(csv_eps):
        df_eps = pd.read_csv(csv_eps)
        
        plt.figure(figsize=(9, 5.5))
        plt.bar(df_eps["epsilon"].astype(str), df_eps["avg_waiting_time_sec"], color="#808080", edgecolor="black", width=0.55)
        plt.title("Replication of Paper Fig. 3: Waiting Time / Delay vs Epsilon Values", fontweight="bold")
        plt.xlabel("Epsilon (ε) Values")
        plt.ylabel("Average Service Delay / Waiting Time (seconds)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        for i, val in enumerate(df_eps["avg_waiting_time_sec"]):
            plt.text(i, val + 0.15, f"{val:.1f}s", ha="center", fontsize=9)
            
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "plots", "paper_fig3_epsilon_sensitivity.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f" Saved Paper Figure 3 replication to {plot_path}")


def generate_all_plots(save_dir: str = "results"):
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    plot_training_curves(save_dir)
    plot_multi_batch_comparison(save_dir)
    plot_ablation_study(save_dir)
    plot_paper_figures_replication(save_dir)


if __name__ == "__main__":
    generate_all_plots()
