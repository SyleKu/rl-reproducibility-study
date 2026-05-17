import torch

from src.algorithms.td3.model import DeterministicPolicy, QNetwork


obs_dim = 3
action_dim = 1
batch_size = 32
action_limit = 2.0

policy = DeterministicPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    action_limit=action_limit,
)

q_net = QNetwork(
    obs_dim=obs_dim,
    action_dim=action_dim,
)

obs = torch.randn(batch_size, obs_dim)
action = policy(obs)
q_value = q_net(obs, action)

print("Action shape:", action.shape)
print("Q value shape:", q_value.shape)
print("Action min:", action.min().item())
print("Action max:", action.max().item())