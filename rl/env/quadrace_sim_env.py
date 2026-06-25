"""ROS-free deploy/eval env for a MonoRace PPO policy, backed by the pure-C++
multirotor_pysim.BatchSim. No Aerostack2, no ROS, no clock/services: each step
sets the motor command and integrates a FIXED dt deterministically
(BatchSim.step(dt)), exactly the policy's 0.01 s training rate.

Public interface matches QuadRaceDeployEnv so rl.scripts.run_quadrace_policy can
drive either backend (reset(start_gate=), step(action) -> obs,0,False,False,info
with info['pos_enu'], set_target_gate, target_gate_idx).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    from gym import spaces

from . import quadrace_adapter as qa
from . import sim_params
from .track import GATES_ENU  # noqa: F401  (re-exported for drivers)

# Make the built pybind11 module importable.
_SIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'sim', 'multirotor_pysim')
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import multirotor_pysim as mps  # noqa: E402


class QuadRaceSimEnv:
    def __init__(self, gates_enu, uav_yaml='config/uav_config_cvar_racing.yaml',
                 *, dt=0.01, start_dist=2.0, ground_z=0.0, floor_z=0.0,
                 w_min=341.75, w_max=3100.0, k=0.5, motor_perm=(0, 1, 2, 3)):
        self._gates_enu = [dict(g) for g in gates_enu]
        self._n_gates = len(self._gates_enu)
        self._dt = float(dt)
        self._start_dist = float(start_dist)
        self._ground_z = float(ground_z)
        self._floor_z = float(floor_z)
        self._wmin, self._wmax, self._k = float(w_min), float(w_max), float(k)
        self._motor_perm = np.asarray(motor_perm, dtype=int)
        self._inv_perm = np.argsort(self._motor_perm)

        self._sim = mps.BatchSim(1)
        self._sim.set_model_params(sim_params.build_params(uav_yaml, num_envs=1))

        # their-frame gate tables + looped relative-gate lookahead
        gp = np.array([qa.vec_enu_to_their([g['x'], g['y'], g['z']])
                       for g in self._gates_enu], dtype=np.float64)
        gy = np.array([qa.yaw_enu_to_their(g['yaw'])
                       for g in self._gates_enu], dtype=np.float64)
        self._gp_their, self._gy_their = gp, gy
        self._gpr, self._gyr = qa.compute_relative_gates(gp, gy)

        self._target_gate_idx = 0
        self._step_idx = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # ---- track / target --------------------------------------------------- #
    def set_target_gate(self, idx):
        self._target_gate_idx = int(idx) % self._n_gates

    @property
    def target_gate_idx(self):
        return self._target_gate_idx

    # ---- state helpers ---------------------------------------------------- #
    def _state(self):
        return self._sim.get_state()[0]  # (17,) ENU/FLU

    def _set_full_state(self, pos, vel, yaw, omega, motors):
        qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        row = np.zeros((1, 17))
        row[0, 0:3] = pos
        row[0, 3:6] = vel
        row[0, 6], row[0, 9] = qw, qz       # level, yaw-only quaternion
        row[0, 10:13] = omega
        row[0, 13:17] = motors
        self._sim.set_state(row)

    # ---- observation (their 20-D) ---------------------------------------- #
    def _get_obs(self):
        s = self._state()
        pos_t = qa.vec_enu_to_their(s[0:3])
        vel_t = qa.vec_enu_to_their(s[3:6])
        r_wb = qa.quat_to_rotmat(s[6], s[7], s[8], s[9])
        quat_t = qa.rotmat_to_quat(qa.rotmat_enu_to_their(r_wb))
        omega_t = qa.body_rates_enu_to_their(s[10:13])
        motor_norm = qa.motor_speed_to_norm(s[13:17][self._motor_perm])
        idx = self._target_gate_idx
        nxt = (idx + 1) % self._n_gates
        obs = qa.assemble_obs_their_frame(
            pos_t, vel_t, quat_t, omega_t, motor_norm,
            self._gp_their[idx], self._gy_their[idx],
            self._gpr[nxt], self._gyr[nxt])
        return obs.astype(np.float32)

    # ---- gym API ---------------------------------------------------------- #
    def reset(self, *, seed=None, start_gate=0):
        self.set_target_gate(start_gate)
        g = self._gates_enu[self._target_gate_idx]
        start_pos = np.array([
            g['x'] - self._start_dist * math.cos(g['yaw']),
            g['y'] - self._start_dist * math.sin(g['yaw']),
            self._ground_z,
        ])
        self._set_full_state(start_pos, np.zeros(3), float(g['yaw']),
                             np.zeros(3), np.zeros(4))
        self._step_idx = 0
        return self._get_obs(), {'target_gate_idx': self._target_gate_idx,
                                 'start_pos': start_pos}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        wc_their = qa.action_to_motor_speed(a, self._wmin, self._wmax, self._k)
        wc_ours = wc_their[self._inv_perm]
        self._sim.set_motor_command(wc_ours.reshape(1, 4))
        self._sim.step(self._dt)
        self._floor_clamp()
        self._step_idx += 1
        s = self._state()
        info = {'pos_enu': s[0:3].copy(), 'yaw_enu': self._yaw_of(s),
                'quat_enu': s[6:10].copy(),  # (qw,qx,qy,qz) body->world FLU/ENU
                'target_gate_idx': self._target_gate_idx, 'step': self._step_idx}
        return self._get_obs(), 0.0, False, False, info

    def close(self):
        pass

    # ---- floor (pure Dynamics has no ground collision) ------------------- #
    def _floor_clamp(self):
        s = self._sim.get_state()
        if s[0, 2] < self._floor_z:
            s[0, 2] = self._floor_z
            s[0, 5] = max(0.0, s[0, 5])  # no downward velocity through floor
            self._sim.set_state(s)

    @staticmethod
    def _yaw_of(s):
        return math.atan2(2 * (s[6] * s[9] + s[7] * s[8]),
                          1 - 2 * (s[8] * s[8] + s[9] * s[9]))
