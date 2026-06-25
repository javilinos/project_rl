"""Train PPO on the MonoRace gate-racing problem in OUR pure-C++ sim
(QuadRaceVecEnv over multirotor_pysim.BatchSim) — the ROS-free, fixed-dt
counterpart to training in the TU-Delft JAX env. Same 20-D obs / 4-motor action,
so the resulting policy is plug-compatible with the JAX env and the deploy env
(QuadRaceSimEnv / run_quadrace_policy --backend sim).

    python -m rl.scripts.train_sim --name SIM1 --num-envs 100
    python -m rl.scripts.train_sim --name SIM1_sde --sde      # gSDE exploration

Watch: tensorboard --logdir rl/logs_sim ; checkpoints -> rl/models_sim/<name>/.
"""

from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from rl.env.quadrace_vec_env import QuadRaceVecEnv

# Domain randomization over OUR model params (field: +/- percent). Mirrors the
# spirit of their wide ModelParams DR, mapped to our field names; tune freely.
DEFAULT_DR = dict(mass=10, inertia=30, thrust_coeff=20, torque_coeff=20,
                  rotor_drag=40, body_quad=40, thrust_k_angle=50, thrust_k_hor=50,
                  time_constant=40, min_speed=20, max_speed=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flight-plan',
                    default='/root/as2_projects/quadrace-diffsim/quadrace-diffsim/flight_plans/test_track.json')
    ap.add_argument('--uav-yaml', default='config/uav_config_cvar_racing.yaml')
    ap.add_argument('--name', default='SIM1')
    ap.add_argument('--num-envs', type=int, default=100)
    ap.add_argument('--steps', type=int, default=int(3e8))
    ap.add_argument('--sde', action='store_true', help='gSDE exploration')
    ap.add_argument('--no-dr', action='store_true', help='disable domain randomization')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    env = VecMonitor(QuadRaceVecEnv(
        args.flight_plan, num_envs=args.num_envs, uav_yaml=args.uav_yaml,
        dr=None if args.no_dr else DEFAULT_DR, seed=0))

    logs, models = 'rl/logs_sim', f'rl/models_sim/{args.name}'
    os.makedirs(models, exist_ok=True)
    model = PPO(
        'MlpPolicy', env, verbose=1, tensorboard_log=logs, device=args.device,
        policy_kwargs=dict(activation_fn=torch.nn.Tanh,
                           net_arch=[dict(pi=[64, 64, 64], vf=[256, 256, 256])],
                           log_std_init=-2 if args.sde else 0),
        n_steps=2000, batch_size=5000, n_epochs=10, gamma=0.999, ent_coef=0.0,
        use_sde=args.sde, sde_sample_freq=8 if args.sde else -1)

    save_every = model.n_steps * env.num_envs * 10
    while model.num_timesteps < args.steps:
        model.learn(save_every, reset_num_timesteps=False, tb_log_name=args.name)
        path = f'{models}/{model.num_timesteps}'
        model.save(path)
        print('saved', path)


if __name__ == '__main__':
    main()
