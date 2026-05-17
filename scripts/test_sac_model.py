import torch

from src.algorithms.sac.model import GaussianPolicy, QNetwork


obs_dim = 3
action_dim = 1
batch_size = 32

policy = GaussianPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    action_limit=2.0,
)

q_net = QNetwork(
    obs_dim=obs_dim,
    action_dim=action_dim,
)

obs = torch.randn(batch_size, obs_dim)

action, log_prob = policy.sample(obs)
q_value = q_net(obs, action)

print("Action shape:", action.shape)
print("Log prob shape:", log_prob.shape)
print("Q value shape:", q_value.shape)

print("Action min:", action.min().item())
print("Action max:", action.max().item())