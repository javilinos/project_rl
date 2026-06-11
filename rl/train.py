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
from rl.env.subproc_swarm_vec_env import make_subproc_swarm_vec_env
from rl.env.swarm_vec_env import make_swarm_vec_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to rl_config.yaml (defaults to rl/rl_config.yaml)')
    parser.add_argument('--run-name', type=str, default='ppo')
    parser.add_argument(
        '--namespaces', type=str, default=None,
        help='Comma-separated drone namespaces, e.g. "drone0,drone1,drone2". '
             'Defaults to drone.namespaces from the config file.')
    parser.add_argument(
        '--resume-from', type=str, default=None,
        help='Path to a .zip PPO checkpoint to continue training from. When '
             'set, the existing weights / optimizer state / hyperparameters '
             'baked into the checkpoint are kept; only the env is rebuilt '
             'and `--timesteps` more steps are run. Skips the fresh PPO(...) '
             'constructor below, so the train.py PPO knobs are NOT applied '
             'on resume — change them in the checkpoint or use a fresh run.')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else here / 'rl_config.yaml'
    models_dir = here / 'models'
    logs_dir = here / 'logs'
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.namespaces:
        namespaces = [ns.strip() for ns in args.namespaces.split(',') if ns.strip()]
    else:
        drone_cfg = cfg.get('drone', {})
        namespaces = drone_cfg.get('namespaces') or [drone_cfg.get('namespace', 'drone0')]
    training_cfg = cfg.get('training', {})
    backend = training_cfg.get('vec_env', 'subproc')
    base_ros_domain_id = training_cfg.get('base_ros_domain_id')
    print(f'training with {len(namespaces)} parallel drone(s) via '
          f'{backend} backend: {namespaces} '
          f'(base_ros_domain_id={base_ros_domain_id})')

    if backend == 'subproc':
        swarm = make_subproc_swarm_vec_env(
            config_path=str(cfg_path),
            namespaces=namespaces,
            base_ros_domain_id=base_ros_domain_id,
        )
    elif backend == 'dummy':
        swarm = make_swarm_vec_env(
            config_path=str(cfg_path), namespaces=namespaces)
    else:
        raise ValueError(
            f"Unknown training.vec_env={backend!r}. "
            'Use "subproc" or "dummy".')
    vec = VecMonitor(swarm)

    if args.resume_from:
        print(f'resuming from checkpoint: {args.resume_from}')
        model = PPO.load(args.resume_from, env=vec,
                         tensorboard_log=str(logs_dir))
    else:
        model = PPO(
            policy='MlpPolicy',
            env=vec,
            verbose=1,
            tensorboard_log=str(logs_dir),
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gae_lambda=0.95,
            normalize_advantage=True,
            # Tuned for action.mode=rates AT control_hz=100 Hz. Every PPO
            # knob that's "counted in steps" got rescaled when the env step
            # rate doubled — same sim-time semantics, twice the step count
            # per sim second:
            #   gamma 0.999 → 0.9995: effective horizon 1/(1−γ) goes 1000
            #     → 2000 steps = 20 s of sim time at 100 Hz (was 20 s at
            #     50 Hz).
            #   sde_sample_freq 12 → 24: noise weights θ persist for 24 ×
            #     10 ms = 240 ms — same sim-time commit window as the
            #     50 Hz value of 12 × 20 ms. Long enough for the policy to
            #     actually see the consequence of a sustained exploration
            #     direction without committing to a tumble.
            # Other knobs (vs the speed-mode defaults) are action-mode
            # tuning, not control-rate tuning:
            #   learning_rate 1e-4 → 3e-5: tiny moves in rate-policy space
            #     change behavior much more than the same delta in
            #     velocity-policy space, so slower updates protect the
            #     "don't tumble" prior once found.
            #   log_std_init: matched to action.rates limits so noise std
            #     ≈ 0.5–0.7 rad/s on rates (survivable for early training).
            # If you flip action.mode back to "speed" OR change control_hz,
            # revisit the matching set together — they're tied.
            gamma=0.9995,
            vf_coef=0.5,
            learning_rate=3e-5,
            ent_coef=0.001,
            use_sde=True,
            sde_sample_freq=24,
            policy_kwargs=dict(
                log_std_init=-3.2,
                ortho_init=True,
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
            )
        )

    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=str(models_dir),
        name_prefix=f'{args.run_name}_drone',
    )
    pause_cb = PhysicsPauseCallback()

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
