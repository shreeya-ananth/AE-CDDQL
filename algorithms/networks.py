"""
Deep Q-Network architecture for CDDQL and AE-CDDQL.
Matches Section 4.1 of the baseline paper:
'The deep Q-network in the CDDQL algorithm contains a sequential model
with 4 dense layers, an input, and an output layer along with ReLU activation
with Adam optimization.'
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            # Dense Layer 1
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            # Dense Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Dense Layer 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Dense Layer 4
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output Layer (Q-value per action)
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass estimating Q(s, a) for all actions.
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            Tensor of shape (batch_size, action_dim)
        """
        return self.network(state)
