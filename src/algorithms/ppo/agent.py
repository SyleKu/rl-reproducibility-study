import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.algorithms.ppo.model import PolicyNetwork, ValueNetwork

class PPOAgent:
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            learning_rate: float = 3e-4,
            clip_epsilon: float = 0.2,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            device: str = "cpu"
    ) -> None:
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

        self.optimizer = Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

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

    def update(self, batch: dict[str, torch.Tensor], ppo_epochs: int) -> dict[str, float]:
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_total_loss = 0.0

        for _ in range(ppo_epochs):
            action_dist = self.policy(observations)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            values = self.value(observations)

            ratio = torch.exp(new_log_probs - old_log_probs)

            unclipped_objective = ratio * advantages
            clipped_objective = (
                torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * advantages
            )

            policy_loss = -torch.min(
                unclipped_objective,
                clipped_objective
            ).mean()

            value_loss = F.mse_loss(values, returns)

            total_loss = (
                policy_loss
                + value_loss * self.value_coef
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = entropy.item()
            last_total_loss = total_loss.item()

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "total_loss": last_total_loss,
        }
