"""PPO training entry point for the drone goal-reaching task.

Usage:
    python -m rl.train [--timesteps 500000]

Requires the patched Aerostack2 simulator to be running (./launch_as2.bash).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from rl.callbacks import PhysicsPauseCallback
from rl.env.swarm_vec_env import make_swarm_vec_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=3_000_000)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to rl_config.yaml (defaults to rl/rl_config.yaml)')
    parser.add_argument('--run-name', type=str, default='ppo')
    parser.add_argument(
        '--namespaces', type=str, default=None,
        help='Comma-separated drone namespaces, e.g. "drone0,drone1,drone2". '
             'Defaults to drone.namespaces from the config file.')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else here / 'rl_config.yaml'
    models_dir = here / 'models'
    logs_dir = here / 'logs'
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    if args.namespaces:
        namespaces = [ns.strip() for ns in args.namespaces.split(',') if ns.strip()]
    else:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        drone_cfg = cfg.get('drone', {})
        namespaces = drone_cfg.get('namespaces') or [drone_cfg.get('namespace', 'drone0')]
    print(f'training with {len(namespaces)} parallel drone(s): {namespaces}')

    swarm = make_swarm_vec_env(config_path=str(cfg_path), namespaces=namespaces)
    vec = VecMonitor(swarm)

    model = PPO(
        policy='MlpPolicy',
        env=vec,
        verbose=1,
        tensorboard_log=str(logs_dir),
        n_steps=1024,
        batch_size=16,
        n_epochs=5,
        gae_lambda=0.93,
        normalize_advantage=True,
        gamma=0.999,
        vf_coef=0.5,
        learning_rate=3e-5,
        ent_coef=0.001,
        use_sde=True,
        sde_sample_freq=8,
        policy_kwargs=dict(
            log_std_init=-1,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )
    )

    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=str(models_dir),
        name_prefix=f'{args.run_name}_drone',
    )
    pause_cb = PhysicsPauseCallback(namespaces=namespaces)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[ckpt_cb, pause_cb],
            tb_log_name=args.run_name,
        )
        model.save(str(models_dir / f'{args.run_name}_drone_final'))
    finally:
        swarm.close()


if __name__ == '__main__':
    main()
