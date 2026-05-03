from dataclasses import dataclass, field
from typing import List

import torch

@dataclass
class PPOBuffer:
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float ] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: float, done: bool, log_prob: torch.Tensor, value: torch.Tensor) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
