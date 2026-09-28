"""
Baseline Clipped Double Deep Q-Learning (CDDQL) Scheduler.
Strictly implements the algorithm described in the baseline paper:
'Task Scheduling in Cloud Using Deep Reinforcement Learning' (Procedia Computer Science 2021).
Key characteristics of the baseline:
1. Two Q-networks: Q1 (actor/evaluator) and Q2 (target/evaluator)
2. CDDQL target calculation: y = r + gamma * min(Q1_targ(s', a*), Q2_targ(s', a*))
   where a* = argmax_a Q1(s', a)
3. Fixed epsilon = 0.5 (as empirically validated and chosen in the baseline paper)
4. 3-dimensional state space: (CPU, Memory, Storage)
5. Target network smoothing via tau = 0.01 and periodic sync
"""
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base_scheduler import BaseScheduler
from algorithms.networks import QNetwork
from algorithms.replay_buffer import ReplayBuffer


class BaselineCDDQLScheduler(BaseScheduler):
    def __init__(
        self,
        state_dim: int = 3, # Baseline: (CPU, Memory, Storage)
        num_vms: int = 4,   # 4 VMs
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        fixed_epsilon: float = 0.5, # Paper Section 4.2 & 5: fixed at 0.5
        tau: float = 0.01,          # Paper Equation (4): tau = 0.01
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 20,
        device: str = "cpu"
    ):
        super().__init__(name="CDDQL-Baseline", num_vms=num_vms)
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = fixed_epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.step_count = 0
        
        # Primary networks Q1 and Q2
        self.q1 = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        
        # Target networks Q1_target and Q2_target
        self.q1_target = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, num_vms, hidden_dim).to(self.device)
        
        # Initialize target networks with same weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers (Adam as specified in Section 4.1)
        self.optimizer1 = optim.Adam(self.q1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.q2.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity, device=device)

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        """
        Algorithm-3: epsilon-greedy policy.
        In baseline CDDQL, epsilon is fixed at 0.5.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: random action in range [0, possible_actions - 1]
            return np.random.randint(self.num_vms)
        else:
            # Exploitation: argmax_a Q1(s, a)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q1.eval()
            with torch.no_grad():
                q_vals = self.q1(state_tensor)
                action = int(torch.argmax(q_vals, dim=1).item())
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
        """Add transition tuple to replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Train Q1 and Q2 networks using CDDQL Bellman target update.
        Algorithm-2 steps 3.12 - 3.16 and Equation (3).
        """
        if len(self.memory) < self.batch_size:
            return None
            
        self.step_count += 1
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calculate CDDQL target (Equation 3):
        # a* = argmax_a' Q1(s', a')
        with torch.no_grad():
            next_q1_vals = self.q1(next_states)
            best_next_actions = torch.argmax(next_q1_vals, dim=1, keepdim=True)
            
            # Evaluate with both target networks and take the minimum (clipped double Q)
            target_q1_val = self.q1_target(next_states).gather(1, best_next_actions)
            target_q2_val = self.q2_target(next_states).gather(1, best_next_actions)
            min_target_q = torch.min(target_q1_val, target_q2_val)
            
            target_y = rewards + (1.0 - dones) * self.gamma * min_target_q
            
        # Update Q1
        current_q1 = self.q1(states).gather(1, actions)
        loss1 = self.criterion(current_q1, target_y)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        
        # Update Q2
        current_q2 = self.q2(states).gather(1, actions)
        loss2 = self.criterion(current_q2, target_y)
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        
        # Soft update target networks: theta' = tau * theta + (1 - tau) * theta' (Equation 4)
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        # Periodic hard sync as in Alg 2 line 3.16
        if self.step_count % self.target_update_freq == 0:
            self.q2_target.load_state_dict(self.q1.state_dict())
            
        return {
            "loss1": float(loss1.item()),
            "loss2": float(loss2.item()),
            "epsilon": self.epsilon
        }
