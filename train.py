"""Train PPO on Gymnasium LunarLander-v3, with periodic rendered previews."""
import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

ENV_ID = "LunarLander-v3"


class PreviewCallback(BaseCallback):
    def __init__(self, interval: int, max_steps: int = 1000):
        super().__init__()
        self.interval = interval
        self.max_steps = max_steps
        self.next_preview = interval
        self.preview_env = None

    def _on_training_start(self):
        self.preview_env = gym.make(ENV_ID, render_mode="human")

    def _on_step(self):
        if self.num_timesteps < self.next_preview:
            return True
        self.next_preview += self.interval
        obs, _ = self.preview_env.reset(seed=42)
        total_reward = 0.0
        for _ in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.preview_env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        print(f"Preview at {self.num_timesteps:,} steps: reward={total_reward:.1f}")
        return True

    def _on_training_end(self):
        if self.preview_env is not None:
            self.preview_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--preview-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    args = parser.parse_args()
    if args.steps <= 0 or args.preview_every < 0 or args.checkpoint_every <= 0:
        parser.error("steps and checkpoint-every must be positive; preview-every must be nonnegative")

    Path("checkpoints").mkdir(exist_ok=True)
    env = gym.make(ENV_ID)
    try:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="logs")
        callbacks = [CheckpointCallback(save_freq=max(1, args.checkpoint_every // model.get_env().num_envs), save_path="checkpoints", name_prefix="ppo_lander")]
        if args.preview_every:
            callbacks.append(PreviewCallback(args.preview_every))
        model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks))
        model.save("lunar_lander_ai")
        print("Saved model: lunar_lander_ai.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
