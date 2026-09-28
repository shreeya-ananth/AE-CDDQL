"""
Task representation for cloud scheduling simulation.
Tasks are modeled after the Google Task Events dataset as used in the paper.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CloudTask:
    task_id: int
    size_mb: float                  # Task size in Megabytes (e.g. 431, 213, 228, 248 MB)
    cpu_req: float                  # CPU processing needed (MB or million instructions)
    memory_req: float               # RAM requirement (213 - 431 MB as specified in paper)
    storage_req: float              # Storage requirement in MB
    arrival_time: float = 0.0       # Arrival timestamp (seconds)
    deadline: Optional[float] = None # SLA deadline
    
    # Execution tracking
    scheduled_vm: Optional[int] = None
    start_time: float = 0.0
    finish_time: float = 0.0
    waiting_time: float = 0.0
    execution_time: float = 0.0
    energy_consumed: float = 0.0    # Joules (W * s)
    cost: float = 0.0               # Monetary cost ($)
    
    def reset(self):
        """Reset execution metrics for new episode evaluation."""
        self.scheduled_vm = None
        self.start_time = 0.0
        self.finish_time = 0.0
        self.waiting_time = 0.0
        self.execution_time = 0.0
        self.energy_consumed = 0.0
        self.cost = 0.0
