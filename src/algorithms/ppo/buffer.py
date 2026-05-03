from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class PPOBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def store(self, obs: np.ndarray, action: int, reward: float, done: bool, log_prob: float, value: float) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0) -> None:
        advantages = []
        gae = 0.0

        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[step])

            delta = (
                self.rewards[step]
                + gamma * values[step + 1] * mask
                - values[step]
            )

            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)

        self.advantages = advantages
        self.returns = [
            advantage + value for advantage, value in zip(self.advantages, values)
        ]

    def get_tensors(self) -> dict[str, torch.Tensor]:
        advantages = torch.tensor(self.advantages, dtype=torch.float32)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "observations": torch.tensor(np.array(self.observations), dtype=torch.float32),
            "actions": torch.tensor(self.actions, dtype=torch.float32),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "advantages": advantages,
            "returns": torch.tensor(self.returns, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
        }

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
