import yaml
import numpy as np
import gymnasium as gym

from src.utils.seed import set_seed
from src.utils.logger import CSVLogger

from src.algorithms.ppo.agent import PPOAgent
from src.algorithms.ppo.buffer import PPOBuffer

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_policy(env: gym.Env, agent: PPOAgent, buffer: PPOBuffer, max_steps: int, seed: int) -> tuple[float, int]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.store(
            obs=np.array(obs, dtype=np.float32),
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        total_reward += float(reward)
        steps += 1
        obs = next_obs

        if done:
            break

    return total_reward, steps

def main():
    config = load_config("configs/ppo.yaml")

    env_name = config["env_name"]
    seed = config["seed"]
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps_per_episode"]
    log_dir = config["log_dir"]
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]

    set_seed(seed)

    env = gym.make(env_name)

    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Only Box observation spaces supported")

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError("Only discrete action spaces supported")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
    buffer = PPOBuffer()
    logger = CSVLogger(log_dir=log_dir)

    rewards = []

    for episode in range(1, num_episodes + 1):
        buffer.clear()

        reward, steps = run_policy(env=env, agent=agent, buffer=buffer, max_steps=max_steps, seed=seed+episode)
        rewards.append(reward)

        buffer.compute_advantages(
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
        )

        batch = buffer.get_tensors()

        rewards.append(reward)
        logger.log(episode=episode, reward=reward, steps=steps)

        mean_reward = np.mean(rewards[-10:])

        print(
            f"Episode: {episode}/{num_episodes} | "
            f"Reward: {reward:.2f} | "
            f"Steps: {steps} | "
            f"Mean (Last 10): {mean_reward:.2f} | "
            f"Buffer: {len(buffer.rewards)} | "
            f"Adv Mean: {batch['advantages'].mean():.4f} | "
            f"Return Mean: {batch['returns'].mean():.4f}"
        )

    env.close()


if __name__ == "__main__":
    main()
