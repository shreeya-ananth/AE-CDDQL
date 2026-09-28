"""
Cloud Environment simulating Datacenter, Virtual Machines, and Task Dispatching.
Implements the exact specifications from 'Task Scheduling in Cloud Using Deep Reinforcement Learning'
and supports the proposed AE-CDDQL state representations and multi-objective rewards.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from simulation.task import CloudTask
from simulation.vm import VirtualMachine
from simulation.workload_generator import generate_paper_tasks, generate_multi_batch_workload


class CloudSimEnvironment:
    """
    Simulated Cloud Datacenter Environment for non-preemptive task scheduling.
    """
    def __init__(
        self,
        num_vms: int = 4,
        state_mode: str = "baseline", # 'baseline' (3-dim) or 'energy_aware' (5-dim)
        reward_mode: str = "baseline", # 'baseline' or 'multi_objective'
        alpha: float = 0.35, # Weight for Execution Time in MO reward
        beta: float = 0.35,  # Weight for Waiting Time in MO reward
        gamma_weight: float = 0.30 # Weight for Energy in MO reward
    ):
        self.num_vms = num_vms
        self.state_mode = state_mode
        self.reward_mode = reward_mode
        self.alpha = alpha
        self.beta = beta
        self.gamma_weight = gamma_weight
        
        # Initialize the 4 Virtual Machines with heterogeneous compute & power tiers
        # Reflecting Section 4.1 of the baseline paper
        self.vms: List[VirtualMachine] = self._create_default_vms()
        
        # Ranking of VMs by computation efficiency for baseline reward:
        # VM with best computation capability -> rank 0 (reward +2)
        # 2nd best -> rank 1 (reward +1)
        # 3rd best -> rank 2 (penalty -1)
        # slowest -> rank 3 (penalty -2)
        self.vm_speed_ranks = self._compute_vm_ranks()
        
        self.tasks: List[CloudTask] = []
        self.current_task_idx: int = 0
        self.current_sim_time: float = 0.0
        
        # Max normalization constants for bounded state and reward scaling
        self.max_size_mb = 500.0
        self.max_mem_mb = 500.0
        self.max_storage_mb = 1000.0
        self.max_workload_mb = 2500.0
        self.max_energy_joules = 10000.0
        self.max_exec_time = 30.0
        self.max_wait_time = 60.0
        
    def _create_default_vms(self) -> List[VirtualMachine]:
        """
        Creates 4 VMs with 500MB RAM and 1000MB Storage as per paper Section 4.1.
        Computation speeds correspond to the 4 efficiency categories:
        VM 0: Best (100 MB/s)
        VM 1: 2nd-Best (75 MB/s)
        VM 2: 3rd-Best (50 MB/s)
        VM 3: Slowest (25 MB/s)
        Power consumption models: P_run and P_idle in Watts
        """
        return [
            VirtualMachine(vm_id=0, processing_speed=100.0, memory_capacity=500.0, storage_capacity=1000.0,
                           p_idle=80.0, p_run=220.0, cost_per_sec=0.00008),
            VirtualMachine(vm_id=1, processing_speed=75.0,  memory_capacity=500.0, storage_capacity=1000.0,
                           p_idle=60.0, p_run=175.0, cost_per_sec=0.00006),
            VirtualMachine(vm_id=2, processing_speed=50.0,  memory_capacity=500.0, storage_capacity=1000.0,
                           p_idle=45.0, p_run=130.0, cost_per_sec=0.00004),
            VirtualMachine(vm_id=3, processing_speed=25.0,  memory_capacity=500.0, storage_capacity=1000.0,
                           p_idle=30.0, p_run=90.0,  cost_per_sec=0.00002),
        ]
        
    def _compute_vm_ranks(self) -> Dict[int, int]:
        """Rank VMs in descending order of processing speed (0 = fastest, 3 = slowest)."""
        sorted_by_speed = sorted(self.vms, key=lambda v: v.processing_speed, reverse=True)
        return {vm.vm_id: rank for rank, vm in enumerate(sorted_by_speed)}

    def reset(self, tasks: Optional[List[CloudTask]] = None) -> np.ndarray:
        """
        Reset datacenter VMs and task queue to start a fresh scheduling episode.
        Algorithm-1 line 12.1: 'Reset the environment to initial state'.
        """
        for vm in self.vms:
            vm.reset()
            
        if tasks is not None:
            self.tasks = tasks
        else:
            self.tasks = generate_paper_tasks(batch_id=0, shuffle=True)
            
        for t in self.tasks:
            t.reset()
            
        self.current_task_idx = 0
        self.current_sim_time = 0.0
        
        return self._get_state()

    @property
    def state_dim(self) -> int:
        """Return dimensionality of the observation space."""
        if self.state_mode == "baseline":
            # Baseline CDDQL state: (CPU, Memory, Storage) -> 3
            return 3
        elif self.state_mode == "energy_aware":
            # Proposed AE-CDDQL state: (CPU, Memory, Storage, Workload, Energy) -> 5
            return 5
        else:
            raise ValueError(f"Unknown state_mode: {self.state_mode}")

    def _get_state(self) -> np.ndarray:
        """
        Calculate the current state representation.
        Algorithm-2 line 3.3.2:
          - Baseline: sk = (CPU, Memory, Storage)
          - Proposed AE-CDDQL: sk = (CPU, Memory, Storage, Workload, Energy)
        """
        if self.current_task_idx >= len(self.tasks):
            # Terminal state padding
            if self.state_mode == "baseline":
                return np.zeros(3, dtype=np.float32)
            else:
                return np.zeros(5, dtype=np.float32)
                
        curr_task = self.tasks[self.current_task_idx]
        
        # Normalized task resource requirements
        norm_cpu = float(curr_task.cpu_req / self.max_size_mb)
        norm_mem = float(curr_task.memory_req / self.max_mem_mb)
        norm_storage = float(curr_task.storage_req / self.max_storage_mb)
        
        if self.state_mode == "baseline":
            # 3-dim baseline state: (CPU, Memory, Storage)
            state = np.array([norm_cpu, norm_mem, norm_storage], dtype=np.float32)
        elif self.state_mode == "energy_aware":
            # 5-dim proposed energy-aware state: (CPU, Memory, Storage, Workload, Energy)
            # Aggregate system workload & energy footprint
            total_workload = sum(vm.current_workload for vm in self.vms)
            total_energy = sum(vm.total_energy for vm in self.vms)
            
            norm_workload = min(1.0, float(total_workload / self.max_workload_mb))
            norm_energy = min(1.0, float(total_energy / self.max_energy_joules))
            
            state = np.array([norm_cpu, norm_mem, norm_storage, norm_workload, norm_energy], dtype=np.float32)
        else:
            raise ValueError(f"Unknown state_mode: {self.state_mode}")
            
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes an action: schedules current task to selected VM (action).
        Algorithm-2 steps 3.4 - 3.8.
        
        Returns:
            next_state (np.ndarray): observation after scheduling
            reward (float): scalar reward for the transition
            done (bool): whether all tasks have been scheduled
            info (dict): execution diagnostic metrics
        """
        if self.current_task_idx >= len(self.tasks):
            raise IndexError("Step called on already finished environment.")
            
        if not (0 <= action < self.num_vms):
            raise ValueError(f"Invalid VM action {action}. Must be in [0, {self.num_vms - 1}].")
            
        task = self.tasks[self.current_task_idx]
        selected_vm = self.vms[action]
        
        # Advance simulation time to task arrival if current time is behind
        self.current_sim_time = max(self.current_sim_time, task.arrival_time)
        
        # Schedule task on selected VM non-preemptively
        exec_info = selected_vm.schedule_task(task, self.current_sim_time)
        
        # Advance current task pointer
        self.current_task_idx += 1
        done = (self.current_task_idx >= len(self.tasks))
        
        # Calculate reward
        reward = self._calculate_reward(action, exec_info)
        
        # Observe next state
        next_state = self._get_state()
        
        info = {
            "task_id": task.task_id,
            "vm_id": action,
            "exec_time": exec_info["exec_time"],
            "wait_time": exec_info["wait_time"],
            "energy": exec_info["energy"],
            "cost": exec_info["cost"],
            "finish_time": exec_info["finish_time"]
        }
        
        return next_state, reward, done, info

    def _calculate_reward(self, action: int, exec_info: dict) -> float:
        """
        Calculates reward according to either the Baseline paper or Proposed Multi-Objective formulation.
        
        1. Baseline Paper reward (Section 4.1 & Section 3.2):
           Section 4.1:
           - Best VM: +2
           - Second-best VM: +1
           - Third-best VM: -1
           - Slowest VM: -2
           Combined with delay penalty:
           Immediate_Reward = +2, +1, -1, -2 (or 1 / (norm_exec + norm_wait))
           
        2. Proposed Multi-Objective Reward:
           Considering: Execution Time + Waiting Time + Energy Consumption
           R_mo = - [ alpha * norm(T_exec) + beta * norm(T_wait) + gamma * norm(E_task) ]
           Normalized into an effective scale so gradients are stable and well-conditioned.
        """
        if self.reward_mode == "baseline":
            # Paper Section 4.1 discrete ranking rewards:
            rank = self.vm_speed_ranks[action]
            if rank == 0:
                reward = 2.0
            elif rank == 1:
                reward = 1.0
            elif rank == 2:
                reward = -1.0
            else: # slowest
                reward = -2.0
            return reward
            
        elif self.reward_mode == "multi_objective":
            # Proposed project modification:
            # Multi-objective reward considering: Execution Time + Waiting Time + Energy Consumption
            norm_exec = exec_info["exec_time"] / self.max_exec_time
            norm_wait = exec_info["wait_time"] / self.max_wait_time
            norm_energy = exec_info["energy"] / 1000.0 # scale Joules
            
            # Penalty formulation: minimizing exec_time, wait_time, and energy
            penalty = (
                self.alpha * norm_exec +
                self.beta * norm_wait +
                self.gamma_weight * norm_energy
            )
            # Center and scale reward around 0
            reward = -penalty
            return float(reward)
            
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics over the completed episode:
        - Makespan (Max completion time)
        - Total & Average Execution Time
        - Total & Average Waiting Time
        - Total Energy Consumption (Joules and kWh)
        - Total Monetary Cost ($)
        - VM Resource Utilization (%)
        """
        if not self.tasks:
            return {}
            
        makespan = max((vm.available_time for vm in self.vms), default=0.0)
        total_exec_time = sum(t.execution_time for t in self.tasks)
        avg_exec_time = total_exec_time / len(self.tasks)
        total_wait_time = sum(t.waiting_time for t in self.tasks)
        avg_wait_time = total_wait_time / len(self.tasks)
        total_energy = sum(vm.total_energy for vm in self.vms)
        total_cost = sum(vm.total_cost for vm in self.vms)
        
        # VM Utilization: busy_time / makespan
        vm_utilizations = []
        for vm in self.vms:
            if makespan > 0:
                u = min(1.0, vm.total_busy_time / makespan)
            else:
                u = 0.0
            vm_utilizations.append(u)
        avg_utilization = float(np.mean(vm_utilizations))
        
        return {
            "num_tasks": len(self.tasks),
            "makespan": makespan,
            "total_execution_time": total_exec_time,
            "avg_execution_time": avg_exec_time,
            "total_waiting_time": total_wait_time,
            "avg_waiting_time": avg_wait_time,
            "total_energy_joules": total_energy,
            "total_energy_kwh": total_energy / (3600.0 * 1000.0),
            "total_cost": total_cost,
            "avg_vm_utilization": avg_utilization,
            "vm_utilizations": vm_utilizations
        }
