from dataclasses import dataclass, field
from collections import deque
import random

import numpy as np
import torch


@dataclass
class SACReplayBuffer:
    capacity: int
    observations: deque = field(init=False)
    actions: deque = field(init=False)
    rewards: deque = field(init=False)
    next_observations: deque = field(init=False)
    dones: deque = field(init=False)

    def __post_init__(self) -> None:
        self.observations = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.next_observations = deque(maxlen=self.capacity)
        self.dones = deque(maxlen=self.capacity)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.observations.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_observations.append(np.array(next_obs, dtype=np.float32))
        self.dones.append(bool(done))

    def sample_batch(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        if len(self) < batch_size:
            raise ValueError(
                f"Cannot sample batch of size {batch_size}; "
                f"buffer only contains {len(self)} transitions."
            )

        indices = random.sample(range(len(self)), batch_size)

        observations = np.array([self.observations[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        next_observations = np.array([self.next_observations[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices], dtype=np.float32)

        return {
            "observations": torch.tensor(observations, dtype=torch.float32, device=device),
            "actions": torch.tensor(actions, dtype=torch.float32, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
            "next_observations": torch.tensor(next_observations, dtype=torch.float32, device=device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1),
        }

    def __len__(self) -> int:
        return len(self.observations)