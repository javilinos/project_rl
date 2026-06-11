"""Curriculum-training driver: bump v_max each time reward plateaus.

Loads an existing PPO checkpoint, runs `model.learn` in stages. Each stage:

  1. Pushes the current v_max into every env (via ``env_method('set_v_max')``).
  2. Trains until either ``--stage-max-steps`` is exhausted OR a `PlateauStopper`
     callback detects that the mean episode reward has been flat for a while.
  3. Saves a checkpoint named after the v_max of the just-finished stage
     ("before bumping" as requested).
  4. Increments v_max by ``--vmax-step`` (default 1 m/s) and starts the next
     stage. Stops when v_max would exceed ``--target-vmax``.

Usage:
    python -m rl.curriculum_train \\
        --initial-checkpoint rl/models/ppo_drone_final.zip \\
        --target-vmax 6.0

A typical curriculum from v_max=2 to v_max=6 with the defaults will produce
checkpoints `curriculum_v2.0_drone.zip`, `curriculum_v3.0_drone.zip`, …,
`curriculum_v6.0_drone.zip`, plus the periodic ``CheckpointCallback`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from rl.callbacks import PhysicsPauseCallback, PlateauStopper
from rl.env.subproc_swarm_vec_env import make_subproc_swarm_vec_env
from rl.env.swarm_vec_env import make_swarm_vec_env


def _build_vec(cfg: dict, cfg_path: Path, namespaces: list[str]):
    training_cfg = cfg.get('training', {})
    backend = training_cfg.get('vec_env', 'subproc')
    base_ros_domain_id = training_cfg.get('base_ros_domain_id')
    print(f'curriculum_train: {len(namespaces)} drone(s) via {backend} backend: '
          f'{namespaces} (base_ros_domain_id={base_ros_domain_id})')
    if backend == 'subproc':
        swarm = make_subproc_swarm_vec_env(
            config_path=str(cfg_path),
            namespaces=namespaces,
            base_ros_domain_id=base_ros_domain_id,
        )
    elif backend == 'dummy':
        swarm = make_swarm_vec_env(str(cfg_path), namespaces)
    else:
        raise ValueError(f'Unknown training.vec_env: {backend!r}')
    return swarm, VecMonitor(swarm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-checkpoint', type=str, required=True,
                        help='Path to the PPO .zip checkpoint to start from.')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--namespaces', type=str, default=None,
                        help='Comma-separated drone namespaces. '
                             'Defaults to drone.namespaces in the config file.')
    parser.add_argument('--run-name', type=str, default='curriculum')

    # Curriculum knobs
    parser.add_argument('--initial-vmax', type=float, default=None,
                        help='Starting v_max. Defaults to action.v_max in the '
                             'config file.')
    parser.add_argument('--target-vmax', type=float, default=6.0,
                        help='Stop once v_max passes this value.')
    parser.add_argument('--vmax-step', type=float, default=1.0,
                        help='Increment v_max by this each stage.')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Optional override for PPO learning_rate. By '
                             'default, the value baked into the checkpoint is '
                             'used unchanged.')

    # Plateau / stage limits
    parser.add_argument('--stage-min-steps', type=int, default=200_000,
                        help='Minimum steps per stage before plateau may fire.')
    parser.add_argument('--stage-max-steps', type=int, default=1_000_000,
                        help='Hard cap on steps per stage.')
    parser.add_argument('--plateau-window', type=int, default=20,
                        help='Plateau detection window (samples). '
                             'window * sample_every is the "flat" period.')
    parser.add_argument('--plateau-sample-every', type=int, default=10_000)
    parser.add_argument('--plateau-min-improve', type=float, default=1.0)

    parser.add_argument('--ckpt-save-freq', type=int, default=50_000,
                        help='CheckpointCallback save frequency within each stage.')
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

    # Start v_max from CLI override or the config.
    current_vmax = (args.initial_vmax
                    if args.initial_vmax is not None
                    else float(cfg['action']['v_max']))
    target_vmax = float(args.target_vmax)
    step = float(args.vmax_step)

    print(f'curriculum: {len(namespaces)} drones, v_max {current_vmax:.2f} → '
          f'{target_vmax:.2f} m/s in {step:.2f} m/s steps')
    print(f'loading checkpoint: {args.initial_checkpoint}')

    swarm, vec = _build_vec(cfg, cfg_path, namespaces)
    model = PPO.load(args.initial_checkpoint, env=vec)
    if args.learning_rate is not None:
        model.learning_rate = args.learning_rate
        model._setup_lr_schedule()
        print(f'overriding learning_rate → {args.learning_rate}')

    try:
        stage = 0
        while current_vmax <= target_vmax + 1e-6:
            print(f'\n=== Stage {stage}: v_max = {current_vmax:.2f} m/s ===')

            # Push v_max into every env. set_v_max also auto-scales depth_tol.
            vec.env_method('set_v_max', current_vmax)

            ckpt_cb = CheckpointCallback(
                save_freq=args.ckpt_save_freq,
                save_path=str(models_dir),
                name_prefix=f'{args.run_name}_v{current_vmax:.1f}_stage{stage}',
            )
            plateau_cb = PlateauStopper(
                stage_max_steps=args.stage_max_steps,
                stage_min_steps=args.stage_min_steps,
                sample_every=args.plateau_sample_every,
                window_size=args.plateau_window,
                min_improve=args.plateau_min_improve,
            )
            # Pause /clock at every rollout boundary so the simulator
            # doesn't drift while PPO is doing its gradient update. Without
            # this, sim time keeps advancing during model.learn's update
            # phase and the next rollout starts against a stale platform
            # state. One instance is fine across all stages — it has no
            # per-stage state.
            pause_cb = PhysicsPauseCallback()

            target_total = model.num_timesteps + args.stage_max_steps
            model.learn(
                total_timesteps=target_total,
                callback=[ckpt_cb, plateau_cb, pause_cb],
                tb_log_name=f'{args.run_name}_v{current_vmax:.1f}',
                reset_num_timesteps=False,
            )

            # Save the checkpoint BEFORE bumping v_max — this freezes the
            # policy at the version that mastered the current speed.
            ckpt_path = models_dir / f'{args.run_name}_v{current_vmax:.1f}_drone.zip'
            model.save(str(ckpt_path))
            print(f'  saved {ckpt_path}  (num_timesteps={model.num_timesteps})')

            current_vmax += step
            stage += 1

        final_path = models_dir / f'{args.run_name}_drone_final.zip'
        model.save(str(final_path))
        print(f'\ncurriculum complete. final policy: {final_path}')
    finally:
        swarm.close()


if __name__ == '__main__':
    main()
