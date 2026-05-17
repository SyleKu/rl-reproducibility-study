import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_limit: float = 1.0,
    ) -> None:
        super().__init__()

        self.action_limit = action_limit

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_action = mean
        else:
            pre_tanh_action = normal.rsample()

        tanh_action = torch.tanh(pre_tanh_action)
        action = tanh_action * self.action_limit

        log_prob = normal.log_prob(pre_tanh_action)
        log_prob -= torch.log(
            self.action_limit * (1 - tanh_action.pow(2)) + EPSILON
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


