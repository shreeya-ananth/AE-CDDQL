"""
Classical baseline schedulers for cloud task scheduling:
1. Random Scheduling (used as baseline in the paper)
2. Round Robin Scheduling (cyclic allocation)
3. Space-Shared / Greedy Earliest Finish (Min-Min heuristic)
4. Minimum Workload Scheduler (load-balancing heuristic)
"""
import random
from typing import Optional, Any
import numpy as np

from algorithms.base_scheduler import BaseScheduler


class RandomScheduler(BaseScheduler):
    """
    Random Selection Scheduler.
    Described in Section 2 & 4.2 of the baseline paper.
    """
    def __init__(self, num_vms: int = 4, seed: Optional[int] = None):
        super().__init__(name="Random", num_vms=num_vms)
        self.rng = random.Random(seed)

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        return self.rng.randint(0, self.num_vms - 1)


class RoundRobinScheduler(BaseScheduler):
    """
    Round Robin Scheduler.
    Cyclically assigns tasks to available VMs in sequence.
    """
    def __init__(self, num_vms: int = 4):
        super().__init__(name="RoundRobin", num_vms=num_vms)
        self.current_idx = 0

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        action = self.current_idx % self.num_vms
        self.current_idx += 1
        return action

    def reset(self):
        self.current_idx = 0


class GreedyEarliestFinishScheduler(BaseScheduler):
    """
    Greedy Earliest-Finish-Time (EFT) / Space-Shared Scheduler.
    Assigns each task to the VM that will complete it at the earliest timestamp,
    factoring in current VM queue availability and processing speed.
    """
    def __init__(self, num_vms: int = 4):
        super().__init__(name="GreedyEFT", num_vms=num_vms)

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        if env is None or env.current_task_idx >= len(env.tasks):
            return 0
        task = env.tasks[env.current_task_idx]
        best_vm = 0
        best_finish = float("inf")
        for vm in env.vms:
            finish = vm.estimate_finish_time(task, env.current_sim_time)
            if finish < best_finish:
                best_finish = finish
                best_vm = vm.vm_id
        return best_vm


class MinWorkloadScheduler(BaseScheduler):
    """
    Minimum Workload Scheduler.
    Selects the VM with the minimum pending workload to achieve basic load balancing.
    """
    def __init__(self, num_vms: int = 4):
        super().__init__(name="MinWorkload", num_vms=num_vms)

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        if env is None:
            return 0
        workloads = [vm.current_workload for vm in env.vms]
        return int(np.argmin(workloads))
