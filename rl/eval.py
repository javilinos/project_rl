"""Load a trained PPO policy and run deterministic rollouts.

Usage:
    python -m rl.eval --model rl/models/ppo_drone_final.zip [--episodes 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.env.drone_goal_env import DroneGoalEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else here / 'rl_config.yaml'

    env = DroneGoalEnv(config_path=str(cfg_path))
    try:
        model = PPO.load(args.model, env=env)
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1
            success = bool(info.get('success'))
            successes += int(success)
            print(
                f'episode={ep} steps={steps} reward={ep_reward:.2f} '
                f'success={success} dist={info.get("dist_xyz", float("nan")):.3f} '
                f'yaw_err={info.get("yaw_err", float("nan")):.3f}'
            )
        print(f'success rate: {successes}/{args.episodes}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
