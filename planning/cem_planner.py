import numpy as np
import torch


class CEMPlanner:
    """
    Cross-Entropy Method planner for continuous actions.
    Plans action sequences using a learned dynamics model.
    """

    def __init__(
        self,
        dynamics_model,
        horizon=20,
        num_samples=512,
        num_elites=64,
        num_iters=5,
        action_low=-1.0,
        action_high=1.0,
        device="cpu",
    ):
        self.model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.action_low = action_low
        self.action_high = action_high
        self.device = device

    @torch.no_grad()
    def plan(self, state, reward_fn):
        """
        state: np.ndarray shape (state_dim,)
        reward_fn: function(state_tensor) -> reward tensor (batch,)
        """
        state0 = torch.tensor(state, dtype=torch.float32, device=self.device)
        state0 = state0.unsqueeze(0).repeat(self.num_samples, 1)

        mean = torch.zeros(self.horizon, device=self.device)
        std = torch.ones(self.horizon, device=self.device)

        for _ in range(self.num_iters):
            actions = mean + std * torch.randn(
                self.num_samples, self.horizon, device=self.device
            )
            actions = torch.clamp(actions, self.action_low, self.action_high)

            states = state0.clone()
            rewards = torch.zeros(self.num_samples, device=self.device)

            for t in range(self.horizon):
                a = actions[:, t:t+1]
                states = self.model(states, a)
                rewards += reward_fn(states)
            
            # terminal bonus
            rewards += 2.0 * reward_fn(states)

            elite_idxs = torch.topk(rewards, self.num_elites).indices
            elite_actions = actions[elite_idxs]

            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean[0].item()
