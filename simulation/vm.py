"""
Virtual Machine representation for cloud scheduling simulation.
Matches parameters specified in Section 3.1 & 4.1 of the baseline paper:
- 4 VMs with 500MB RAM and 1000MB Storage
- Processing efficiency tiers (best, 2nd-best, 3rd-best, slowest)
- Idle & Running power models (PIdle, PRun)
- Operational cost modeling
"""
from dataclasses import dataclass, field
from typing import List
from simulation.task import CloudTask


@dataclass
class VirtualMachine:
    vm_id: int
    processing_speed: float      # Computation efficiency in MB/s
    memory_capacity: float = 500.0   # 500 MB RAM as in paper
    storage_capacity: float = 1000.0 # 1000 MB Storage as in paper
    p_idle: float = 50.0         # Idle power in Watts (P_idle)
    p_run: float = 150.0         # Running power in Watts (P_run)
    cost_per_sec: float = 0.00005 # Cost per second ($)
    
    # Dynamic operational state
    available_time: float = 0.0  # Timestamp when VM becomes free
    current_workload: float = 0.0 # Sum of pending task sizes (MB)
    tasks_queue: List[CloudTask] = field(default_factory=list)
    
    # Cumulative statistics
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    total_energy: float = 0.0    # Joules (W * s)
    total_cost: float = 0.0
    tasks_executed: int = 0
    
    def reset(self):
        """Reset the VM state for a new simulation episode."""
        self.available_time = 0.0
        self.current_workload = 0.0
        self.tasks_queue.clear()
        self.total_busy_time = 0.0
        self.total_idle_time = 0.0
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.tasks_executed = 0
        
    def estimate_finish_time(self, task: CloudTask, current_sim_time: float) -> float:
        """Estimate completion time if task is assigned to this VM."""
        start_time = max(self.available_time, current_sim_time)
        exec_time = task.size_mb / self.processing_speed
        return start_time + exec_time

    def schedule_task(self, task: CloudTask, current_sim_time: float) -> dict:
        """
        Execute task allocation on this VM non-preemptively.
        Updates VM operational timeline, energy, and costs.
        """
        start_time = max(self.available_time, current_sim_time)
        wait_time = max(0.0, start_time - task.arrival_time)
        exec_time = task.size_mb / self.processing_speed
        finish_time = start_time + exec_time
        
        # Energy consumption:
        # Running energy during active computation + Idle energy during waiting gap
        idle_duration = max(0.0, start_time - self.available_time)
        energy_idle = self.p_idle * idle_duration
        energy_active = self.p_run * exec_time
        task_energy = energy_active + energy_idle
        
        # Monetary cost
        task_cost = self.cost_per_sec * (exec_time + idle_duration)
        
        # Update task execution records
        task.scheduled_vm = self.vm_id
        task.start_time = start_time
        task.finish_time = finish_time
        task.waiting_time = wait_time
        task.execution_time = exec_time
        task.energy_consumed = task_energy
        task.cost = task_cost
        
        # Update VM state
        self.total_idle_time += idle_duration
        self.total_busy_time += exec_time
        self.total_energy += task_energy
        self.total_cost += task_cost
        self.available_time = finish_time
        self.tasks_executed += 1
        self.tasks_queue.append(task)
        self.current_workload += task.size_mb
        
        return {
            "vm_id": self.vm_id,
            "start_time": start_time,
            "finish_time": finish_time,
            "wait_time": wait_time,
            "exec_time": exec_time,
            "energy": task_energy,
            "cost": task_cost
        }
