"""
Adaptive Energy-Aware Clipped Double Deep Q-Learning (AE-CDDQL) for Multi-Objective Cloud Task Scheduling.

Implements the proposed project modifications over the baseline CDDQL:
1. Adaptive epsilon-greedy instead of fixed epsilon (0.5)
2. Energy-aware state: (CPU, Memory, Storage, Workload, Energy) -> 5 dimensions
3. Multi-objective reward considering: Execution Time + Waiting Time + Energy Consumption

Includes modular flags for individual ablations:
- CDDQL (Baseline)
- CDDQL + Adaptive Epsilon
- CDDQL + Energy State
- CDDQL + Multi-Objective Reward
- Full AE-CDDQL
"""
from typing import Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base_scheduler import BaseScheduler
from algorithms.networks import QNetwork
from algorithms.replay_buffer import ReplayBuffer


class AECDQLScheduler(BaseScheduler):
    def __init__(
        self,
        state_dim: int = 5,             # 5 for proposed energy-aware state; 3 for baseline
        num_vms: int = 4,               # 4 Virtual Machines
        hidden_dim: int = 128,          # 4 dense layers hidden dimension
        lr: float = 1e-3,
        gamma: float = 0.99,            # Discount factor from paper
        use_adaptive_epsilon: bool = True, # Modification 1
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.005,    # Exponential decay parameter
        fixed_epsilon: float = 0.5,     # Used when adaptive_epsilon is False (baseline)
        tau: float = 0.01,              # Target network update rate (Equation 4)
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 20,
        device: str = "cpu"
    ):
        name = "AE-CDDQL" if (use_adaptive_epsilon and state_dim == 5) else "CDDQL-Variant"
        super().__init__(name=name, num_vms=num_vms)
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.step_count = 0
        self.episode_count = 0
        
        # Modification 1: Adaptive vs Fixed Epsilon
        self.use_adaptive_epsilon = use_adaptive_epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.fixed_epsilon = fixed_epsilon
        
        if self.use_adaptive_epsilon:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = self.fixed_epsilon
            
        # Neural Networks: 4 Dense Layers each (Section 4.1)
        self.q1 = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers (Adam optimization with MSE loss)
        self.optimizer1 = optim.Adam(self.q1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.q2.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Replay memory
        self.memory = ReplayBuffer(capacity=buffer_capacity, device=device)

    def update_epsilon(self, step: Optional[int] = None):
        """
        Adaptive epsilon decay policy (Modification 1).
        Smoothly decays from exploration (1.0) to exploitation (0.02)
        using an exponential schedule based on scheduling steps or episodes.
        """
        if not self.use_adaptive_epsilon:
            self.epsilon = self.fixed_epsilon
            return
            
        current_t = self.step_count if step is None else step
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.epsilon_decay * current_t)

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        """
        Select action using epsilon-greedy strategy with adaptive or fixed epsilon.
        Exploration selects random VM; exploitation queries Q1 network.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_vms)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q1.eval()
            with torch.no_grad():
                q_values = self.q1(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            self.q1.train()
            return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Record transition tuple in experience replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform a gradient descent learning step using Clipped Double Deep Q-Learning.
        CDDQL Bellman target:
          y = r + gamma * (1 - done) * min(Q1_target(s', a*), Q2_target(s', a*))
          where a* = argmax_a' Q1(s', a')
        """
        if len(self.memory) < self.batch_size:
            return None
            
        self.step_count += 1
        
        # Update adaptive epsilon on every step
        if self.use_adaptive_epsilon:
            self.update_epsilon()
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Clipped Double Q-Target computation
        with torch.no_grad():
            next_q1_vals = self.q1(next_states)
            best_actions = torch.argmax(next_q1_vals, dim=1, keepdim=True)
            
            target_q1 = self.q1_target(next_states).gather(1, best_actions)
            target_q2 = self.q2_target(next_states).gather(1, best_actions)
            min_target_q = torch.min(target_q1, target_q2)
            
            y_target = rewards + (1.0 - dones) * self.gamma * min_target_q
            
        # Update Primary Network Q1
        q1_pred = self.q1(states).gather(1, actions)
        loss1 = self.criterion(q1_pred, y_target)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        
        # Update Primary Network Q2
        q2_pred = self.q2(states).gather(1, actions)
        loss2 = self.criterion(q2_pred, y_target)
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        
        # Soft update target networks with tau = 0.01 (Equation 4)
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            
        # Periodic hard synchronization (Algorithm 2 line 3.16)
        if self.step_count % self.target_update_freq == 0:
            self.q2_target.load_state_dict(self.q1.state_dict())
            
        return {
            "loss1": float(loss1.item()),
            "loss2": float(loss2.item()),
            "epsilon": float(self.epsilon)
        }
