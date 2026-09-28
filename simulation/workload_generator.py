"""
Workload generator for cloud task scheduling.
Synthesizes task batches directly mirroring the Google Task Events dataset
as configured in Section 4.1 and Section 4.2 of the baseline paper.
"""
import random
from typing import List
from simulation.task import CloudTask


# Exact task sizes defined in Section 4.2 of the baseline paper:
# Tasks 1 to 4: 431 MB
# Tasks 5 to 6: 213 MB
# Tasks 7 to 8: 228 MB
# Task 9: 248 MB
PAPER_BENCHMARK_TASK_SIZES = [
    431.0, 431.0, 431.0, 431.0,  # Tasks 1-4
    213.0, 213.0,                # Tasks 5-6
    228.0, 228.0,                # Tasks 7-8
    248.0                        # Task 9
]


def generate_paper_tasks(batch_id: int = 0, shuffle: bool = False, seed: int = None) -> List[CloudTask]:
    """
    Generate a batch of 9 tasks with sizes exactly matching the baseline paper's Google Task Events setup.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    tasks = []
    sizes = list(PAPER_BENCHMARK_TASK_SIZES)
    if shuffle:
        rng.shuffle(sizes)
        
    for idx, size in enumerate(sizes):
        task_id = batch_id * 9 + idx + 1
        # In the paper, RAM ranges from 213 to 431 MB
        # We scale memory requirement with task size
        mem_req = float(size)
        storage_req = rng.uniform(100.0, 300.0) # Within 1000MB VM storage limit
        cpu_req = float(size) # CPU demand proportional to size
        # Arrival intervals: tasks arrive sequentially or concurrently
        arrival = batch_id * 5.0 + idx * 0.5
        
        task = CloudTask(
            task_id=task_id,
            size_mb=size,
            cpu_req=cpu_req,
            memory_req=mem_req,
            storage_req=storage_req,
            arrival_time=arrival,
            deadline=arrival + (size / 25.0) * 2.0 # Realistic SLA deadline
        )
        tasks.append(task)
        
    return tasks


def generate_multi_batch_workload(num_batches: int = 5, shuffle: bool = False, seed: int = 42) -> List[CloudTask]:
    """
    Generates multiple batches of 9 tasks each, matching Figure 2 of the paper
    ('Execution time of 5 batches with 9 tasks each').
    """
    all_tasks = []
    rng = random.Random(seed)
    for b in range(num_batches):
        batch_seed = rng.randint(1, 100000)
        tasks = generate_paper_tasks(batch_id=b, shuffle=shuffle, seed=batch_seed)
        all_tasks.extend(tasks)
    return all_tasks


def generate_synthetic_workload(num_tasks: int = 50, seed: int = 42) -> List[CloudTask]:
    """
    Generate an extended synthetic workload sampled from the Google Task Events distribution
    for large-scale stress testing and generalization analysis.
    """
    rng = random.Random(seed)
    tasks = []
    size_choices = [213.0, 228.0, 248.0, 350.0, 431.0]
    
    current_time = 0.0
    for i in range(num_tasks):
        size = rng.choice(size_choices)
        mem = rng.uniform(200.0, 450.0)
        storage = rng.uniform(100.0, 400.0)
        current_time += rng.expovariate(1.0 / 2.0) # Poisson arrival process (mean 2s)
        
        task = CloudTask(
            task_id=i + 1,
            size_mb=size,
            cpu_req=size,
            memory_req=mem,
            storage_req=storage,
            arrival_time=round(current_time, 2),
            deadline=current_time + (size / 25.0) * 2.5
        )
        tasks.append(task)
    return tasks
