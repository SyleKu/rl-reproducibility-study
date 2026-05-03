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

def run_policy(
        env: gym.Env,
        agent: PPOAgent,
        buffer: PPOBuffer,
        max_steps: int,
        seed: int
) -> tuple[float, int]:
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

def evaluate_policy(
        env: gym.Env,
        agent: PPOAgent,
        num_episodes: int,
        max_steps: int,
        seed: int,
) -> float:
    eval_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0

        for _ in range(max_steps):
            action, _, _ = agent.select_action(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        eval_rewards.append(total_reward)

    return float(np.mean(eval_rewards))

def main():
    config = load_config("configs/ppo.yaml")

    env_name = config["env_name"]
    seed = config["seed"]
    max_steps = config["max_steps_per_episode"]
    log_dir = config["log_dir"]
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]

    num_updates = config["num_updates"]
    episodes_per_update = config["episodes_per_update"]

    eval_interval = config["eval_interval"]
    eval_episodes = config["eval_episodes"]
    checkpoint_dir = config["checkpoint_dir"]
    best_eval_reward = float("-inf")

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

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=config["learning_rate"],
        clip_epsilon=config["clip_epsilon"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
    )

    buffer = PPOBuffer()
    logger = CSVLogger(log_dir=log_dir)

    rewards = []
    episode_count = 0

    for update in range(1, num_updates + 1):
        buffer.clear()

        update_rewards = []
        update_steps = 0

        for _ in range(episodes_per_update):
            episode_count += 1

            reward, steps = run_policy(
                env=env,
                agent=agent,
                buffer=buffer,
                max_steps=max_steps,
                seed=seed+episode_count
            )

            rewards.append(reward)
            update_rewards.append(reward)
            update_steps += steps

            logger.log(
                episode=episode_count,
                reward=reward,
                steps=steps
            )

        buffer.compute_advantages(
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        batch = buffer.get_tensors()

        losses = agent.update(
            batch=batch,
            ppo_epochs=config["ppo_epochs"],
            mini_batch_size=config["mini_batch_size"],
        )

        mean_reward = np.mean(rewards[-10:])
        update_mean_reward = np.mean(update_rewards)

        print(
            f"Update: {update}/{num_updates} | "
            f"Episodes: {episode_count} | "
            f"Update Reward Mean: {update_mean_reward:.2f} | "
            f"Mean Reward Last 10: {mean_reward:.2f} | "
            f"Steps Collected: {update_steps} | "
            f"Policy Loss: {losses['policy_loss']:.4f} | "
            f"Value Loss: {losses['value_loss']:.4f} | "
            f"Entropy: {losses['entropy']:.4f}"
        )

        if update % eval_interval == 0:
            eval_reward = evaluate_policy(
                env=env,
                agent=agent,
                num_episodes=eval_episodes,
                max_steps=max_steps,
                seed=seed + 10_000 + update
            )

            print(
                f"Evaluation | Update: {update} | "
                f"Mean Reward: {eval_reward:.2f}"
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward

                checkpoint_path = f"{checkpoint_dir}/ppo_cartpole_best.pt"
                agent.save(checkpoint_path)

                print(
                    f"New best model saved | "
                    f"Eval Reward: {best_eval_reward:.2f} | "
                    f"Path: {checkpoint_path}"
                )

    env.close()


if __name__ == "__main__":
    main()
