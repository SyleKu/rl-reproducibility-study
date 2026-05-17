from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from src.algorithms.td3.model import DeterministicPolicy, QNetwork


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        policy_delay: int = 2,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau
        self.action_limit = action_limit

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.total_updates = 0

        self.actor = DeterministicPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_limit=action_limit,
        ).to(self.device)

        self.actor_target = DeterministicPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_limit=action_limit,
        ).to(self.device)

        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        self.q1_target = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=critic_lr)

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
            action = self.actor(obs_tensor)

        action_np = action.squeeze(0).cpu().numpy()

        if not deterministic:
            noise = np.random.normal(
                loc=0.0,
                scale=self.exploration_noise * self.action_limit,
                size=action_np.shape,
            )
            action_np = action_np + noise

        return np.clip(
            action_np,
            -self.action_limit,
            self.action_limit,
        ).astype(np.float32)

    def soft_update_targets(self) -> None:
        self._soft_update(self.actor, self.actor_target)
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
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
                "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
                "total_updates": self.total_updates,
            },
            checkpoint_path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])

        self.total_updates = checkpoint["total_updates"]
