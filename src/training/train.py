import yaml
import numpy as np
import gymnasium as gym

from src.utils.seed import set_seed
from src.utils.logger import  CSVLogger

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_random_policy(env: gym.Env, max_steps: int) -> tuple[float, int]:
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return total_reward, steps

def main():
    config = load_config("configs/ppo.yaml")

    env_name = config["env_name"]
    seed = config["seed"]
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps_per_episode"]
    log_dir = config["log_dir"]

    set_seed(seed)

    env = gym.make(env_name)
    logger = CSVLogger(log_dir=log_dir)

    rewards = []

    for episode in range(1, num_episodes + 1):
        env.reset(seed=seed + episode)
        reward, steps = run_random_policy(env, max_steps=max_steps)
        rewards.append(reward)

        logger.Log(episode=episode, reward=reward, steps=steps)

        mean_reward = np.mean(rewards[-10:])
        print(
            f"Episode: {episode}/{num_episodes} | "
            f"Reward: {reward:.2f} | Steps: {steps} | "
            f"Mean (Last 10): {mean_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
