"""QuadRaceVecEnv — SB3 VecEnv that trains the MonoRace gate-racing problem on
OUR pure-C++ multirotor sim (multirotor_pysim.BatchSim), ROS-free.

It is a faithful port of quad_race.environment.QuadRace (same 20-D gate-relative
observation, same reward / gate-pass / crash logic, same init distributions and
flight-plan format) with the dynamics backend swapped: instead of their JAX
`f_func` + forward-Euler, each step transforms the (their-frame NED/FRD) state to
our ENU/FLU frame, integrates one fixed dt with BatchSim, and transforms back.
Internal bookkeeping stays in their frame so the ported reward/obs code matches
theirs and a policy trained here is plug-compatible with their JAX env and our
deploy env (QuadRaceSimEnv).

Defaults (env_config) mirror options_test_track.json; flight plan is their
test_track.json (their NED/FRD frame).
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

_SIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'sim', 'multirotor_pysim')
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import multirotor_pysim as mps  # noqa: E402

_T = np.array([1.0, -1.0, -1.0])          # Rx(180) on pos/vel/omega
_QT = np.array([1.0, 1.0, -1.0, -1.0])    # Rx(180) on quat (qw,qx,qy,qz)
W_MAX_N = 3000.0                          # their motor-speed normalization

DEFAULT_ENV_CONFIG = dict(
    speed_limit=99.0, gate_size=0.8, cam_angle_degrees=50.0,
    initialize_on_ground=False, initialize_uniform=True,
    initialize_at_random_gates=False, loop_gates=True,
    ground_height=0.0, gate_thickness=0.5, v_ground=2.0,
    progress_reward=1.0, gate_reward=1.0, angular_rate_penalty=0.001,
    gate_offset_penalty=1.0, perception_penalty=0.01,
    motor_penalty=0.0, motor_penalty_threshold=0.0, low_action_penalty=0.0,
    crash_penalty=10.0, max_steps=2000,
)


def _euler_from_quat(q):
    """Their Tait-Bryan euler (phi,theta,psi) from N x4 quat (qw,qx,qy,qz)."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    theta = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return phi, theta, psi


class QuadRaceVecEnvPy(VecEnv):
    def __init__(self, flight_plan, num_envs=100, dt=0.01,
                 uav_yaml='config/uav_config_cvar_racing.yaml',
                 env_config=None, dr=None, motor_perm=(3, 0, 1, 2),
                 w_min=341.75, w_max=3100.0, k=0.5, seed=None):
        cfg = dict(DEFAULT_ENV_CONFIG)
        if env_config:
            cfg.update(env_config)
        self.cfg = cfg
        self.dt = np.float32(dt)
        self.num_envs = int(num_envs)
        self._rng = np.random.default_rng(seed)
        self._uav_yaml, self._dr = uav_yaml, dr
        self._wmin, self._wmax, self._k = w_min, w_max, k
        self._perm = np.asarray(motor_perm, int)
        self._inv = np.argsort(self._perm)
        self.cam_angle = float(cfg['cam_angle_degrees']) * np.pi / 180.0

        # --- flight plan (their NED/FRD frame) ---
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
        self.gate_pos_rel, self.gate_yaw_rel = qa.compute_relative_gates(
            self.gate_pos, self.gate_yaw)

        # --- spaces (QuadRace defaults: euler attitude, gates_ahead=1 -> 20) ---
        self.state_len = 20
        obs_space = spaces.Box(-np.inf, np.inf, (self.state_len,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        VecEnv.__init__(self, self.num_envs, obs_space, act_space)

        # --- backend + bookkeeping ---
        self.sim = mps.BatchSim(self.num_envs)
        self._params = sim_params.build_params(uav_yaml, self.num_envs, dr,
                                               seed=seed)
        self.sim.set_model_params(self._params)

        self.world_states = np.zeros((self.num_envs, 17), dtype=np.float32)  # their frame
        self.states = np.zeros((self.num_envs, self.state_len), dtype=np.float32)
        self.target_gates = np.zeros(self.num_envs, dtype=np.int32)
        self.step_counts = np.zeros(self.num_envs, dtype=np.int32)
        self.actions = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.dones = np.zeros(self.num_envs, dtype=bool)

    # ================= frame transforms (their <-> batchsim ENU) ============ #
    def _their_to_sim(self, W):
        """N x17 their-frame -> N x17 BatchSim ENU (physical motors, our order)."""
        S = np.zeros_like(W)
        S[:, 0:3] = W[:, 0:3] * _T
        S[:, 3:6] = W[:, 3:6] * _T
        S[:, 6:10] = W[:, 6:10] * _QT
        S[:, 10:13] = W[:, 10:13] * _T
        w_phys_their = (W[:, 13:17] + 1.0) / 2.0 * W_MAX_N      # normalized->rad/s
        S[:, 13:17] = w_phys_their[:, self._inv]               # their order->our order
        return S

    def _sim_to_their(self, S):
        """N x17 BatchSim ENU -> N x17 their-frame (normalized motors, their order)."""
        W = np.zeros_like(S)
        W[:, 0:3] = S[:, 0:3] * _T
        W[:, 3:6] = S[:, 3:6] * _T
        W[:, 6:10] = S[:, 6:10] * _QT
        W[:, 10:13] = S[:, 10:13] * _T
        w_phys_their = S[:, 13:17][:, self._perm]              # our order->their order
        W[:, 13:17] = 2.0 * w_phys_their / W_MAX_N - 1.0
        return W

    # ================= observation (port of update_states) ================= #
    def update_states(self):
        W = self.world_states
        gate_pos = self.gate_pos[self.target_gates % self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates % self.num_gates]
        c, s = np.cos(gate_yaw), np.sin(gate_yaw)
        ns = np.zeros((self.num_envs, self.state_len), dtype=np.float32)
        # pos in gate frame
        dx, dy = W[:, 0] - gate_pos[:, 0], W[:, 1] - gate_pos[:, 1]
        ns[:, 0] = c * dx + s * dy
        ns[:, 1] = -s * dx + c * dy
        ns[:, 2] = W[:, 2] - gate_pos[:, 2]
        # vel in gate frame
        ns[:, 3] = c * W[:, 3] + s * W[:, 4]
        ns[:, 4] = -s * W[:, 3] + c * W[:, 4]
        ns[:, 5] = W[:, 5]
        # euler attitude (psi relative to gate)
        phi, theta, psi = _euler_from_quat(W[:, 6:10])
        psi = (psi - gate_yaw + np.pi) % (2 * np.pi) - np.pi
        ns[:, 6] = (phi + np.pi) % (2 * np.pi) - np.pi
        ns[:, 7] = (theta + np.pi) % (2 * np.pi) - np.pi
        ns[:, 8] = psi
        # rates + rpms
        ns[:, 9:12] = W[:, 10:13]
        ns[:, 12:16] = W[:, 13:17]
        # next gate relative to current gate (loop)
        nxt = (self.target_gates + 1)
        valid = nxt < self.num_gates if not self.cfg['loop_gates'] else np.ones(self.num_envs, bool)
        nxt = nxt % self.num_gates
        ns[valid, 16:19] = self.gate_pos_rel[nxt[valid]]
        ns[valid, 19] = self.gate_yaw_rel[nxt[valid]]
        self.states = ns

    def _perc_angle(self, W, look_gate_pos):
        """Angle (rad) between the body camera axis and the look gate (their FRD)."""
        # optical axis in body frame
        oa = np.array([np.cos(self.cam_angle), 0.0, -np.sin(self.cam_angle)])
        # gate in body frame: R_wb^T (gate - pos), via quaternion conjugate rotate
        q = W[:, 6:10]
        rel = look_gate_pos - W[:, 0:3]
        gb = _rotate_by_quat_conj(q, rel)
        num = gb @ oa
        den = np.linalg.norm(gb, axis=1) * np.linalg.norm(oa) + 1e-9
        return np.arccos(np.clip(num / den, -1.0, 1.0))

    # ================= VecEnv API ========================================== #
    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def reset_(self, dones):
        idx = np.where(dones)[0]
        m = len(idx)
        if m == 0:
            return np.nan_to_num(self.states)
        cfg = self.cfg
        if cfg['initialize_at_random_gates']:
            self.target_gates[idx] = self._rng.integers(0, self.num_gates, m)
            gp = self.gate_pos[self.target_gates[idx] % self.num_gates]
            gy = self.gate_yaw[self.target_gates[idx] % self.num_gates]
            pos = gp - 2 * np.stack([np.cos(gy), np.sin(gy), np.zeros_like(gy)], 1)
            x0, y0, z0 = pos[:, 0], pos[:, 1], pos[:, 2]
        elif cfg['initialize_uniform']:
            x0 = self._rng.uniform(self.bounds_xy[0, 0], self.bounds_xy[0, 1], m)
            y0 = self._rng.uniform(self.bounds_xy[1, 0], self.bounds_xy[1, 1], m)
            z0 = self._rng.uniform(-5, 0, m)
            self.target_gates[idx] = self._closest_front_gate(x0, y0)
        else:
            self.target_gates[idx] = 0
            x0 = np.full(m, self.start_pos[0]); y0 = np.full(m, self.start_pos[1])
            z0 = np.full(m, self.start_pos[2])

        if cfg['initialize_on_ground']:
            x0 = self._rng.uniform(-0.5, 0.5, m) + self.start_pos[0]
            y0 = self._rng.uniform(-0.5, 0.5, m) + self.start_pos[1]
            z0 = np.zeros(m)
            vx0 = vy0 = vz0 = np.zeros(m); phi0 = theta0 = np.zeros(m)
            psi0 = self._rng.uniform(-np.pi / 4, np.pi / 4, m) + \
                self.gate_yaw[self.target_gates[idx] % self.num_gates]
            p0 = q0 = r0 = np.zeros(m)
            w0 = -np.ones((m, 4))
        else:
            vx0 = self._rng.uniform(-0.5, 0.5, m); vy0 = self._rng.uniform(-0.5, 0.5, m)
            vz0 = self._rng.uniform(-0.5, 0.5, m)
            phi0 = self._rng.uniform(-np.pi / 9, np.pi / 9, m)
            theta0 = self._rng.uniform(-np.pi / 9, np.pi / 9, m)
            psi0 = self._rng.uniform(-np.pi, np.pi, m)
            p0 = self._rng.uniform(-0.1, 0.1, m); q0 = self._rng.uniform(-0.1, 0.1, m)
            r0 = self._rng.uniform(-0.1, 0.1, m)
            w0 = self._rng.uniform(-1, 1, (m, 4))

        quat0 = _quat_from_euler(phi0, theta0, psi0)
        W = np.stack([x0, y0, z0, vx0, vy0, vz0,
                      quat0[:, 0], quat0[:, 1], quat0[:, 2], quat0[:, 3],
                      p0, q0, r0, w0[:, 0], w0[:, 1], w0[:, 2], w0[:, 3]], 1).astype(np.float32)
        self.world_states[idx] = W
        self.step_counts[idx] = 0
        # domain randomize the model params of the reset envs, then push to sim
        self._resample_params(idx)
        self.sim.reset_envs(list(idx), self._their_to_sim(W))
        self.update_states()
        return np.nan_to_num(self.states)

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        # ---- dynamics: their state -> ENU BatchSim -> their state ----
        wc_their = qa.action_to_motor_speed(self.actions, self._wmin, self._wmax, self._k)
        self.sim.set_motor_command(np.ascontiguousarray(wc_their[:, self._inv]))
        self.sim.step(float(self.dt))
        new_states = self._sim_to_their(self.sim.get_state()).astype(np.float32)

        self.step_counts += 1
        pos_old = self.world_states[:, 0:3]
        pos_new = new_states[:, 0:3]
        pos_gate = self.gate_pos[self.target_gates % self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates % self.num_gates]
        cfg = self.cfg

        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        prog = d2g_old - d2g_new
        prog[prog > cfg['speed_limit'] * self.dt] = cfg['speed_limit'] * self.dt
        rewards = cfg['progress_reward'] * prog
        rewards -= cfg['angular_rate_penalty'] * np.linalg.norm(new_states[:, 10:13], axis=1)
        # motor penalty (action diff above threshold)
        sa = (self.actions + 1) / 2
        spa = (self.prev_actions + 1) / 2
        rewards -= cfg['low_action_penalty'] * np.sum(np.clip(0.5 - sa, 0, None), axis=1)
        rewards -= cfg['motor_penalty'] * np.sum(
            np.clip(np.abs(sa - spa) - cfg['motor_penalty_threshold'], 0, None), axis=1)
        # perception penalty (only when gate off-axis > 60 deg)
        if cfg['perception_penalty'] > 0:
            look = self.gate_pos[self.target_gates % self.num_gates]
            pa = self._perc_angle(new_states, look)
            mask = pa > np.pi / 3
            rewards[mask] -= cfg['perception_penalty'] * pa[mask]

        # gate plane crossing
        normal = np.stack([np.cos(yaw_gate), np.sin(yaw_gate)], 1)
        proj_old = (pos_old[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (pos_old[:, 1] - pos_gate[:, 1]) * normal[:, 1]
        proj_new = (pos_new[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (pos_new[:, 1] - pos_gate[:, 1]) * normal[:, 1]
        crossed = (proj_old < 0) & (proj_new > 0)
        half = cfg['gate_size'] / 2.0
        chl = np.max(np.abs(pos_new - pos_gate), axis=1)
        gate_passed = crossed & (chl < half)
        gate_collision = crossed & (chl > half)
        # Solid-frame collision: inside ANY gate's frame band (between the inner
        # opening gate_size/2 and the outer 2.7/2), within +/- thickness/2 of the
        # plane. Uses each gate's OWN z (fixes the multi-level ladder, where the
        # upstream target-z check misplaces the boxes vertically).
        half_out = 2.7 / 2.0
        d = cfg['gate_thickness'] / 2.0
        for gp_i, gy_i in zip(self.gate_pos, self.gate_yaw):
            ci, si = np.cos(gy_i), np.sin(gy_i)
            ddx, ddy = pos_new[:, 0] - gp_i[0], pos_new[:, 1] - gp_i[1]
            nrm = ddx * ci + ddy * si
            lat = -ddx * si + ddy * ci
            dz = pos_new[:, 2] - gp_i[2]
            frame = ((np.abs(nrm) < d)
                     & ((np.abs(lat) > half) | (np.abs(dz) > half))
                     & (np.abs(lat) < half_out) & (np.abs(dz) < half_out))
            gate_collision |= frame
        rewards[gate_passed] = cfg['gate_reward'] - cfg['gate_offset_penalty'] * d2g_new[gate_passed] / half
        rewards[gate_collision] = -cfg['crash_penalty']

        ground = (new_states[:, 2] > -cfg['ground_height']) & (np.linalg.norm(new_states[:, 3:6], axis=1) > cfg['v_ground'])
        rewards[ground] = -cfg['crash_penalty']
        oob = np.zeros(self.num_envs, bool)
        if self.bounds_xy is not None:
            oob |= (new_states[:, 0] < self.bounds_xy[0, 0]) | (new_states[:, 0] > self.bounds_xy[0, 1])
            oob |= (new_states[:, 1] < self.bounds_xy[1, 0]) | (new_states[:, 1] > self.bounds_xy[1, 1])
        oob |= new_states[:, 2] < -10
        oob |= np.any(np.abs(new_states[:, 10:13]) > (1700 * np.pi / 180), axis=1)
        rewards[oob] = -cfg['crash_penalty']

        max_reached = self.step_counts >= cfg['max_steps']
        self.target_gates[gate_passed] += 1
        if cfg['loop_gates']:
            self.target_gates %= self.num_gates
            final = np.zeros(self.num_envs, bool)
        else:
            final = self.target_gates >= self.num_gates
        dones = max_reached | ground | gate_collision | oob | final
        self.dones = dones

        self.world_states = new_states
        self.update_states()
        states_ret = self.states.copy()
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]['terminal_observation'] = states_ret[i]
            if max_reached[i]:
                infos[i]['TimeLimit.truncated'] = True
            infos[i]['gate_passed'] = bool(gate_passed[i])
        self.reset_(dones)
        return (np.nan_to_num(self.states), np.nan_to_num(rewards),
                np.nan_to_num(dones), infos)

    # ---- helpers ----
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
                j = np.argmin(dist[i][behind[i]])
                out[i] = np.where(behind[i])[0][j]
        return out

    def _resample_params(self, idx):
        if not self._dr:
            return
        fresh = sim_params.build_params(self._uav_yaml, len(idx), self._dr)
        for key in self._params:
            self._params[key][idx] = fresh[key]
        self.sim.set_model_params(self._params)

    # ---- VecEnv abstract methods ----
    def close(self): pass
    def seed(self, seed=None): self._rng = np.random.default_rng(seed); return [seed]
    def get_attr(self, attr_name, indices=None): raise AttributeError(attr_name)
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, *a, **k): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def render(self, mode='human'): return None


def _quat_from_euler(phi, theta, psi):
    """N quaternions (qw,qx,qy,qz) from Tait-Bryan xyz euler arrays."""
    cphi, sphi = np.cos(phi / 2), np.sin(phi / 2)
    cth, sth = np.cos(theta / 2), np.sin(theta / 2)
    cps, sps = np.cos(psi / 2), np.sin(psi / 2)
    qw = cphi * cth * cps + sphi * sth * sps
    qx = sphi * cth * cps - cphi * sth * sps
    qy = cphi * sth * cps + sphi * cth * sps
    qz = cphi * cth * sps - sphi * sth * cps
    return np.stack([qw, qx, qy, qz], 1)


def _rotate_by_quat_conj(q, v):
    """Rotate world vectors v (N x3) into body frame: R_wb^T v, via q conjugate."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # body = R^T v; build R (body->world) then transpose-apply
    R00 = 1 - 2 * (qy * qy + qz * qz); R01 = 2 * (qx * qy - qz * qw); R02 = 2 * (qx * qz + qy * qw)
    R10 = 2 * (qx * qy + qz * qw); R11 = 1 - 2 * (qx * qx + qz * qz); R12 = 2 * (qy * qz - qx * qw)
    R20 = 2 * (qx * qz - qy * qw); R21 = 2 * (qy * qz + qx * qw); R22 = 1 - 2 * (qx * qx + qy * qy)
    bx = R00 * v[:, 0] + R10 * v[:, 1] + R20 * v[:, 2]
    by = R01 * v[:, 0] + R11 * v[:, 1] + R21 * v[:, 2]
    bz = R02 * v[:, 0] + R12 * v[:, 1] + R22 * v[:, 2]
    return np.stack([bx, by, bz], 1)
