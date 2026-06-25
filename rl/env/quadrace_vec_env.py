"""QuadRaceVecEnv — SB3 VecEnv over the C++ QuadRaceBatch hot path.

The per-step math (dynamics + frame transform + obs + reward + gate/collision)
runs in C++/OpenMP (multirotor_pysim.QuadRaceBatch). This Python shell only does
the episode reset: detect dones, sample init states + DR (numpy RNG, so resets
stay reproducible/tunable), and push them to C++. Same 20-D obs / 4-motor action
as their JAX env and the deploy env — policies are interchangeable.

The pure-Python reference implementation lives in quadrace_vec_env_py.py
(QuadRaceVecEnvPy); rl.scripts.check_quadrace_cpp_parity diffs the two.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from . import quadrace_adapter as qa
from . import sim_params
from .quadrace_vec_env_py import DEFAULT_ENV_CONFIG, _quat_from_euler

_SIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'sim', 'multirotor_pysim')
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import multirotor_pysim as mps  # noqa: E402


class QuadRaceVecEnv(VecEnv):
    def __init__(self, flight_plan, num_envs=100, dt=0.01,
                 uav_yaml='config/uav_config_cvar_racing.yaml',
                 env_config=None, dr=None, motor_perm=(3, 0, 1, 2),
                 w_min=341.75, w_max=3100.0, k=0.5, seed=None):
        cfg = dict(DEFAULT_ENV_CONFIG)
        if env_config:
            cfg.update(env_config)
        self.cfg = cfg
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self._rng = np.random.default_rng(seed)
        self._dr = dr

        # flight plan (their NED/FRD frame)
        fp = json.load(open(flight_plan))
        start = np.array(fp['start'], dtype=np.float32)
        gates = np.array(fp['gates'], dtype=np.float32)
        start[3] *= np.pi / 180.0
        gates[:, 3] *= np.pi / 180.0
        self.start_pos = start[0:3]
        self.gate_pos = gates[:, 0:3].astype(np.float32)
        self.gate_yaw = gates[:, 3].astype(np.float32)
        self.num_gates = self.gate_pos.shape[0]
        self.bounds_xy = (np.array(fp['bounds_xy'], dtype=np.float32)
                          if isinstance(fp.get('bounds_xy'), list) else None)
        gpr, gyr = qa.compute_relative_gates(self.gate_pos, self.gate_yaw)
        self.gate_pos_rel, self.gate_yaw_rel = gpr, gyr

        obs_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        VecEnv.__init__(self, self.num_envs, obs_space, act_space)

        # C++ batch
        self.qrb = mps.QuadRaceBatch(self.num_envs)
        self.qrb.set_track(self.gate_pos.astype(np.float64), self.gate_yaw.astype(np.float64),
                           self.gate_pos_rel.astype(np.float64), self.gate_yaw_rel.astype(np.float64),
                           None if self.bounds_xy is None else self.bounds_xy.astype(np.float64))
        c = dict(cfg)
        c.update(dt=self.dt, w_min=w_min, w_max=w_max, k=k,
                 cam_angle=float(cfg['cam_angle_degrees']) * np.pi / 180.0,
                 loop_gates=1.0 if cfg['loop_gates'] else 0.0,
                 motor_perm=np.asarray(motor_perm, np.float64))
        self.qrb.set_config(c)
        # cache nominal params once (no per-reset YAML read); seed initial models
        self._nominal = sim_params.build_params(uav_yaml, 1)
        self.qrb.set_model_params(self._dr_params(self.num_envs, randomize=False))

        self.actions = np.zeros((self.num_envs, 4), np.float32)

    # ---- DR params from cached nominal (no YAML read per reset) ----------- #
    def _dr_params(self, m, randomize=True):
        out = {}
        for key, base1 in self._nominal.items():
            base = np.repeat(np.asarray(base1, np.float64), m, axis=0)
            if (randomize and self._dr and key in self._dr and self._dr[key]
                    and key != 'motors_direction'):
                pct = self._dr[key]
                base = base * self._rng.uniform(1 - pct / 100, 1 + pct / 100, base.shape)
            out[key] = base
        return out

    # ---- init-distribution sampling (their frame), mirrors QuadRaceVecEnvPy.reset_
    def _sample_inits(self, m):
        cfg = self.cfg
        rng = self._rng
        if cfg['initialize_at_random_gates']:
            tgt = rng.integers(0, self.num_gates, m).astype(np.int32)
            gp = self.gate_pos[tgt % self.num_gates]; gy = self.gate_yaw[tgt % self.num_gates]
            pos = gp - 2 * np.stack([np.cos(gy), np.sin(gy), np.zeros_like(gy)], 1)
            x0, y0, z0 = pos[:, 0], pos[:, 1], pos[:, 2]
        elif cfg['initialize_uniform']:
            x0 = rng.uniform(self.bounds_xy[0, 0], self.bounds_xy[0, 1], m)
            y0 = rng.uniform(self.bounds_xy[1, 0], self.bounds_xy[1, 1], m)
            z0 = rng.uniform(-5, 0, m)
            tgt = self._closest_front_gate(x0, y0)
        else:
            tgt = np.zeros(m, np.int32)
            x0 = np.full(m, self.start_pos[0]); y0 = np.full(m, self.start_pos[1])
            z0 = np.full(m, self.start_pos[2])

        if cfg['initialize_on_ground']:
            x0 = rng.uniform(-0.5, 0.5, m) + self.start_pos[0]
            y0 = rng.uniform(-0.5, 0.5, m) + self.start_pos[1]
            z0 = np.zeros(m)
            vx0 = vy0 = vz0 = np.zeros(m); phi0 = theta0 = np.zeros(m)
            psi0 = rng.uniform(-np.pi / 4, np.pi / 4, m) + self.gate_yaw[tgt % self.num_gates]
            p0 = q0 = r0 = np.zeros(m); w0 = -np.ones((m, 4))
        else:
            vx0 = rng.uniform(-0.5, 0.5, m); vy0 = rng.uniform(-0.5, 0.5, m); vz0 = rng.uniform(-0.5, 0.5, m)
            phi0 = rng.uniform(-np.pi / 9, np.pi / 9, m); theta0 = rng.uniform(-np.pi / 9, np.pi / 9, m)
            psi0 = rng.uniform(-np.pi, np.pi, m)
            p0 = rng.uniform(-0.1, 0.1, m); q0 = rng.uniform(-0.1, 0.1, m); r0 = rng.uniform(-0.1, 0.1, m)
            w0 = rng.uniform(-1, 1, (m, 4))

        quat = _quat_from_euler(phi0, theta0, psi0)
        W = np.stack([x0, y0, z0, vx0, vy0, vz0, quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3],
                      p0, q0, r0, w0[:, 0], w0[:, 1], w0[:, 2], w0[:, 3]], 1).astype(np.float64)
        return W, tgt.astype(np.float64)

    def _closest_front_gate(self, x0, y0):
        m = len(x0)
        dist = np.zeros((m, self.num_gates)); behind = np.zeros((m, self.num_gates), bool)
        for i in range(self.num_gates):
            p, y = self.gate_pos[i], self.gate_yaw[i]
            dist[:, i] = np.hypot(x0 - p[0], y0 - p[1])
            behind[:, i] = (np.cos(y) * (x0 - p[0]) + np.sin(y) * (y0 - p[1])) < 0
        out = np.zeros(m, np.int32)
        for i in range(m):
            if not behind[i].any():
                out[i] = np.argmin(dist[i])
            else:
                j = np.argmin(dist[i][behind[i]]); out[i] = np.where(behind[i])[0][j]
        return out

    # ---- VecEnv API ------------------------------------------------------- #
    def reset(self):
        idx = list(range(self.num_envs))
        W, tg = self._sample_inits(self.num_envs)
        obs = self.qrb.reset_envs(idx, W, tg, self._dr_params(self.num_envs))
        return np.nan_to_num(obs.astype(np.float32))

    def step_async(self, actions):
        self.actions = np.asarray(actions, np.float32)

    def step_wait(self):
        obs, rew, done, passed, trunc = self.qrb.step(self.actions.astype(np.float64))
        done = done > 0.5
        obs = obs.astype(np.float32)
        idx = np.where(done)[0]
        infos = [{} for _ in range(self.num_envs)]
        if idx.size:
            terminal = obs[idx].copy()
            W, tg = self._sample_inits(idx.size)
            robs = self.qrb.reset_envs(list(idx), W, tg, self._dr_params(idx.size))
            obs[idx] = robs.astype(np.float32)
            for k, i in enumerate(idx):
                infos[int(i)]['terminal_observation'] = terminal[k]
                if trunc[i] > 0.5:
                    infos[int(i)]['TimeLimit.truncated'] = True
        for i in range(self.num_envs):
            infos[i]['gate_passed'] = bool(passed[i] > 0.5)
        return (np.nan_to_num(obs), np.nan_to_num(rew.astype(np.float32)),
                np.nan_to_num(done), infos)

    def close(self):
        pass

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed); return [seed]

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, *a, **k):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def render(self, mode='human'):
        return None
