from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim import Adam

from src.algorithms.sac.model import GaussianPolicy, QNetwork


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_limit = action_limit

        self.policy = GaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_limit=action_limit,
        ).to(self.device)

        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        self.q1_target = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_optimizer = Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=critic_lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.policy.sample(
                obs_tensor,
                deterministic=deterministic,
            )

        return action.squeeze(0).cpu().numpy()

    def soft_update_targets(self) -> None:
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    def _soft_update(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
    ) -> None:
        for source_param, target_param in zip(
            source.parameters(),
            target.parameters(),
        ):
            target_param.data.copy_(
                self.tau * source_param.data
                + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
                "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])

        self.policy_optimizer.load_state_dict(
            checkpoint["policy_optimizer_state_dict"]
        )
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer_state_dict"]
        )

    def update(
            self,
            batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        # -------------------------
        # Critic update
        # -------------------------
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_observations)

            target_q1 = self.q1_target(next_observations, next_actions)
            target_q2 = self.q2_target(next_observations, next_actions)
            target_q_min = torch.min(target_q1, target_q2)

            target_q = rewards + self.gamma * (1.0 - dones) * (
                    target_q_min - self.alpha.detach() * next_log_probs
            )

        current_q1 = self.q1(observations, actions)
        current_q2 = self.q2(observations, actions)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # -------------------------
        # Actor update
        # -------------------------
        new_actions, log_probs = self.policy.sample(observations)

        q1_new = self.q1(observations, new_actions)
        q2_new = self.q2(observations, new_actions)
        q_new_min = torch.min(q1_new, q2_new)

        policy_loss = (
                self.alpha.detach() * log_probs - q_new_min
        ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # -------------------------
        # Alpha / entropy update
        # -------------------------
        alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -------------------------
        # Target network update
        # -------------------------
        self.soft_update_targets()

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }
