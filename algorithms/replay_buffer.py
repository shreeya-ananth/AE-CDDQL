"""
Experience Replay Buffer for CDDQL and AE-CDDQL.
Matches Algorithm 1 & 2 in the paper:
Stores transition tuples (s, a, r, s', done) and samples mini-batches.
"""
from collections import deque
import random
from typing import Tuple, List
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, device: str = "cpu"):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Append transition to replay memory."""
        self.buffer.append((
            np.array(state, dtype=np.float32, copy=True),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(done)
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch of transitions uniformly at random."""
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states_tensor = torch.from_numpy(np.stack(states)).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        return len(self.buffer)
