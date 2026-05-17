import numpy as np

from src.algorithms.sac.agent import SACAgent
from src.algorithms.sac.replay_buffer import SACReplayBuffer


obs_dim = 3
action_dim = 1

agent = SACAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    action_limit=2.0,
)

buffer = SACReplayBuffer(capacity=1000)

for _ in range(200):
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = np.random.uniform(-2.0, 2.0, size=(action_dim,)).astype(np.float32)
    reward = float(np.random.randn())
    next_obs = np.random.randn(obs_dim).astype(np.float32)
    done = False

    buffer.store(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done,
    )

batch = buffer.sample_batch(batch_size=64)
losses = agent.update(batch)

print(losses)