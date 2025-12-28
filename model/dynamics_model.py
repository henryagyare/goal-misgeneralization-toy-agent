import torch
import torch.nn as nn


class DynamicsModel(nn.Module):
    """
    Predicts next state given current state and action.

    Input:
        [x, v, m, integrity, a]
    Output:
        [x', v', m', integrity']

    Note:
        - mass is included but should remain invariant
        - model is free to learn that (important for analysis)
    """

    def __init__(self, state_dim=4, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        """
        state: tensor (..., 4)
        action: tensor (..., 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
