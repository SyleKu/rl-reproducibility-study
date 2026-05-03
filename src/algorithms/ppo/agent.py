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

    def update(
            self,
            batch: dict[str, torch.Tensor],
            ppo_epochs: int,
            mini_batch_size: int,
    ) -> dict[str, float]:
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        batch_size = observations.shape[0]

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_total_loss = 0.0

        for _ in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_indices = indices[start:end]

                mb_observations = observations[mini_batch_indices]
                mb_actions = actions[mini_batch_indices]
                mb_old_log_probs = old_log_probs[mini_batch_indices]
                mb_advantages = advantages[mini_batch_indices]
                mb_returns = returns[mini_batch_indices]

                action_dist = self.policy(mb_observations)
                new_log_probs = action_dist.log_prob(mb_actions)
                entropy = action_dist.entropy().mean()

                values = self.value(mb_observations)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                unclipped_objective = ratio * mb_advantages
                clipped_objective = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon
                    ) * mb_advantages
                )

                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective
                ).mean()

                value_loss = F.mse_loss(values, mb_returns)

                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
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
