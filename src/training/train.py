import gym

def main():
    env = gym.make('CartPole-v1')

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Episode reward: {total_reward}")

if __name__ == "__main__":
    main()
