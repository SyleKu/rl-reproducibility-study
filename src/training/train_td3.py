from pathlib import Path

import yaml
import numpy as np
import gymnasium as gym

from src.utils.seed import set_seed
from src.utils.logger import CSVLogger
from src.algorithms.td3.agent import TD3Agent
from src.algorithms.sac.replay_buffer import SACReplayBuffer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_run_dir(base_dir: str) -> Path:
    base_log_dir = Path(base_dir)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(base_log_dir.glob("td3_run_*"))
    run_id = len(existing_runs) + 1

    run_dir = base_log_dir / f"td3_run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

def evaluate_policy(
    env: gym.Env,
    agent: TD3Agent,
    num_episodes: int,
    max_episode_steps: int,
    seed: int,
) -> float:
    eval_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0

        for _ in range(max_episode_steps):
            action = agent.select_action(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        eval_rewards.append(total_reward)

    return float(np.mean(eval_rewards))

def main() -> None:
    config = load_config("configs/td3.yaml")

    env_name = config["env_name"]
    seed = config["seed"]
    total_steps = config["total_steps"]
    start_steps = config["start_steps"]
    update_after = config["update_after"]
    update_every = config["update_every"]
    batch_size = config["batch_size"]
    replay_buffer_capacity = config["replay_buffer_capacity"]
    max_episode_steps = config["max_episode_steps"]

    eval_interval = config["eval_interval"]
    eval_episodes = config["eval_episodes"]
    checkpoint_dir = config["checkpoint_dir"]
    best_eval_reward = float("-inf")

    set_seed(seed)

    run_dir = create_run_dir(config["log_dir"])
    logger = CSVLogger(log_dir=str(run_dir))

    env = gym.make(env_name)

    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Only Box observation spaces supported")

    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_limit = float(env.action_space.high[0])
    else:
        raise ValueError("Only continuous Box action spaces supported")

    agent = TD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=action_limit,
        hidden_dim=config["hidden_dim"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        policy_noise=config["policy_noise"],
        noise_clip=config["noise_clip"],
        exploration_noise=config["exploration_noise"],
        policy_delay=config["policy_delay"],
    )

    replay_buffer = SACReplayBuffer(capacity=replay_buffer_capacity)

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_steps = 0
    episode = 1

    last_losses = {
        "q1_loss": 0.0,
        "q2_loss": 0.0,
        "actor_loss": 0.0,
        "total_updates": 0.0,
    }

    for step in range(1, total_steps + 1):
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        episode_steps += 1
        episode_reward += float(reward)

        timeout = episode_steps >= max_episode_steps
        done = terminated or truncated or timeout

        replay_buffer.store(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )

        obs = next_obs

        if step >= update_after and step % update_every == 0:
            batch = replay_buffer.sample_batch(
                batch_size=batch_size,
                device=str(agent.device),
            )
            last_losses = agent.update(batch)

        if done:
            if step % eval_interval == 0:
                eval_reward = evaluate_policy(
                    env=env,
                    agent=agent,
                    num_episodes=eval_episodes,
                    max_episode_steps=max_episode_steps,
                    seed=seed + 10_000 + step,
                )

                print(
                    f"Evaluation | "
                    f"Step: {step} | "
                    f"Mean Reward: {eval_reward:.2f}"
                )

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward

                    checkpoint_path = f"{checkpoint_dir}/td3_pendulum_best.pt"
                    agent.save(checkpoint_path)

                    print(
                        f"New best TD3 model saved | "
                        f"Eval Reward: {best_eval_reward:.2f} | "
                        f"Path: {checkpoint_path}"
                    )

            logger.log(
                episode=episode,
                reward=episode_reward,
                steps=episode_steps,
            )

            print(
                f"Step: {step}/{total_steps} | "
                f"Episode: {episode} | "
                f"Reward: {episode_reward:.2f} | "
                f"Steps: {episode_steps} | "
                f"Q1 Loss: {last_losses['q1_loss']:.4f} | "
                f"Q2 Loss: {last_losses['q2_loss']:.4f} | "
                f"Actor Loss: {last_losses['actor_loss']:.4f}"
            )

            obs, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            episode_steps = 0
            episode += 1

    env.close()


if __name__ == "__main__":
    main()
