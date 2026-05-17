import numpy as np

from src.algorithms.sac.agent import SACAgent


agent = SACAgent(
    obs_dim=3,
    action_dim=1,
    action_limit=2.0,
)

obs = np.random.randn(3).astype(np.float32)

action = agent.select_action(obs)
deterministic_action = agent.select_action(obs, deterministic=True)

print("Action:", action)
print("Deterministic action:", deterministic_action)
print("Alpha:", agent.alpha.item())