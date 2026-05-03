import numpy as np
import torch

from src.algorithms.ppo.model import PolicyNetwork, ValueNetwork

class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, device: str = "cpu") -> None:
        self.device = torch.device(device)

        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.value = ValueNetwork(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_dist = self.policy(obs_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.value(obs_tensor)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )
