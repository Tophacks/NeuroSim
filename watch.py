"""Watch a trained PPO policy play LunarLander-v3."""
import argparse
import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lunar_lander_ai")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=100 + episode)
            total = 0.0
            for _ in range(1500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            print(f"Episode {episode + 1}: reward={total:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
