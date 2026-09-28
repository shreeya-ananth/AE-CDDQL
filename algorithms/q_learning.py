"""
Tabular Q-Learning Scheduler (QLS).
Implements the baseline Q-Learning method described in Section 2, 3.2, and 4.2 of the paper.
Uses state discretization and a single Q-table with epsilon-greedy action selection.
"""
from typing import Dict, Tuple, Optional, Any
import numpy as np

from algorithms.base_scheduler import BaseScheduler


class QLearningScheduler(BaseScheduler):
    def __init__(
        self,
        num_vms: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.5, # Fixed epsilon=0.5 as reported for QLS in the baseline paper
        num_bins: int = 4
    ):
        super().__init__(name="QLS", num_vms=num_vms)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_bins = num_bins
        # Q-table represented as a dictionary mapping discrete state tuples to action Q-values
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize continuous state features into discrete bins."""
        bins = np.linspace(0.0, 1.0, self.num_bins + 1)[1:-1]
        discrete = tuple(int(np.digitize(x, bins)) for x in state)
        return discrete

    def _get_q_values(self, discrete_state: Tuple[int, ...]) -> np.ndarray:
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_vms, dtype=np.float32)
        return self.q_table[discrete_state]

    def select_action(self, state: np.ndarray, env: Optional[Any] = None) -> int:
        discrete_state = self._discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_vms)
        q_vals = self._get_q_values(discrete_state)
        # Random tie-breaking among max Q-values
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """Standard Bellman equation Q-table update."""
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        
        q_vals_s = self._get_q_values(s)
        current_q = q_vals_s[action]
        
        if done:
            target = reward
        else:
            q_vals_next = self._get_q_values(s_next)
            target = reward + self.gamma * np.max(q_vals_next)
            
        td_error = target - current_q
        q_vals_s[action] += self.alpha * td_error
        
        return {"td_error": float(abs(td_error))}
