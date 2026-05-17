import numpy as np

from src.algorithms.sac.replay_buffer import SACReplayBuffer

buffer = SACReplayBuffer(capacity=1000)

for _ in range(100):
    buffer.store(
        obs=np.random.randn(3),
        action=np.random.randn(1),
        reward=1.0,
        next_obs=np.random.randn(3),
        done=False,
    )

batch = buffer.sample_batch(batch_size=32)

print(batch["observations"].shape)
print(batch["actions"].shape)
print(batch["rewards"].shape)