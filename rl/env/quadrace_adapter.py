"""Frame transform + observation assembly bridging our Aerostack2 sim to the
TU-Delft MonoRace policy interface (`quad_race.environment.QuadRace`).

Why this exists
---------------
We train a PPO baseline in their fast JAX simulator on our race track, then
deploy the resulting policy in our Aerostack2 sim. The two simulators use
different coordinate conventions:

  * Ours (Aerostack2 / ROS REP-103): ENU world (z up), FLU body (x fwd, y left,
    z up), yaw in radians CCW.
  * Theirs (MonoRace): z-down world, FRD body (x fwd, y right, z down), gate yaw
    fed in degrees but stored radians.

The map between them is a single **global Rx(180deg)** rotation
``T = diag(1, -1, -1)`` applied to BOTH the world and the body frame (the
standard ROS<->aerospace ENU/FLU <-> NED/FRD convention). It is a proper
rotation (det +1) and its own inverse, so the drone's handedness — and hence a
trained policy's left/right/roll sense — is preserved.

This module is intentionally PURE (no ROS, no env): it takes plain arrays so it
can be unit-tested against their `QuadRace.update_states` offline. The live
adapter env (`quadrace_deploy_env.py`) feeds it readings from the running sim.

Their observation (with the QuadRace defaults `train.py` uses: ``R6_input=False``,
``gates_ahead=1``, no history/param) is 20-D:

    [0:2]  pos_G xy   (target-gate frame: Rz(-gate_yaw) . (pos - gate))
    [2]    pos_z - gate_z
    [3:5]  vel_G xy   (same Rz on world velocity)
    [5]    vel_z
    [6]    roll  phi
    [7]    pitch theta
    [8]    yaw   psi - gate_yaw            (each euler wrapped to [-pi, pi])
    [9:12] body rates p, q, r
    [12:16] normalized motor speeds w_i in [-1, 1]  (W_phys = (w+1)/2 * 3000)
    [16:19] next gate position, relative to current gate frame
    [19]    next gate yaw, relative to current gate
"""

from __future__ import annotations

import numpy as np

# Their normalization constant for motor-speed state (environment.py:62).
# NOTE: this is 3000, NOT w_max (3100) — the state is normalized to a fixed
# reference so the policy sees the same scale regardless of the randomized w_max.
W_MAX_N = 3000.0

# Global Rx(180): ENU/FLU -> NED/FRD. Diagonal, self-inverse.
_T_DIAG = np.array([1.0, -1.0, -1.0])
_T_MAT = np.diag(_T_DIAG)


# --------------------------------------------------------------------------- #
# Frame transform (ENU/FLU  ->  their NED/FRD)
# --------------------------------------------------------------------------- #
def vec_enu_to_their(v) -> np.ndarray:
    """Transform a world position OR world velocity (x,y,z) -> (x,-y,-z)."""
    return np.asarray(v, dtype=np.float64) * _T_DIAG


def yaw_enu_to_their(yaw: float) -> float:
    """ENU yaw (CCW from +x) -> their yaw. Under Rx(180): psi -> -psi."""
    return -float(yaw)


def body_rates_enu_to_their(omega) -> np.ndarray:
    """FLU body rates (p,q,r) -> FRD body rates: (p, -q, -r)."""
    return np.asarray(omega, dtype=np.float64) * _T_DIAG


def rotmat_enu_to_their(r_wb) -> np.ndarray:
    """Body->world rotation in their frame: R' = T R T  (T = diag(1,-1,-1))."""
    r_wb = np.asarray(r_wb, dtype=np.float64)
    return _T_MAT @ r_wb @ _T_MAT


# --------------------------------------------------------------------------- #
# Attitude helpers (their conventions)
# --------------------------------------------------------------------------- #
def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    """Body->world rotation matrix from a quaternion (qw, qx, qy, qz)."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = q / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def rotmat_to_quat(r) -> np.ndarray:
    """Body->world rotation matrix -> quaternion (qw, qx, qy, qz), w>=0."""
    r = np.asarray(r, dtype=np.float64)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def their_euler_from_quat(q) -> np.ndarray:
    """Their Tait-Bryan euler (environment.py:56-58) from (qw,qx,qy,qz):
    returns (phi, theta, psi) = (roll, pitch, yaw)."""
    qw, qx, qy, qz = (float(v) for v in q)
    phi = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    theta = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    psi = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array([phi, theta, psi], dtype=np.float64)


def _wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Relative-gate table (mirror of QuadRace.__init__ lines 438-454)
# --------------------------------------------------------------------------- #
def compute_relative_gates(gate_pos, gate_yaw):
    """Per-gate position/yaw of gate i expressed in gate (i-1)'s frame.
    `gate_pos` (N,3) and `gate_yaw` (N,) are in THEIR frame. Looped track."""
    gate_pos = np.asarray(gate_pos, dtype=np.float64)
    gate_yaw = np.asarray(gate_yaw, dtype=np.float64)
    n = gate_pos.shape[0]
    pos_rel = np.zeros((n, 3), dtype=np.float64)
    yaw_rel = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d = gate_pos[i] - gate_pos[i - 1]
        c, s = np.cos(gate_yaw[i - 1]), np.sin(gate_yaw[i - 1])
        # R = [[c, s], [-s, c]] (world->gate of previous gate)
        pos_rel[i, 0] = c * d[0] + s * d[1]
        pos_rel[i, 1] = -s * d[0] + c * d[1]
        pos_rel[i, 2] = d[2]
        yaw_rel[i] = _wrap_pi(gate_yaw[i] - gate_yaw[i - 1])
    return pos_rel, yaw_rel


# --------------------------------------------------------------------------- #
# Observation assembly (THEIR frame in, 20-D obs out)
# --------------------------------------------------------------------------- #
def assemble_obs_their_frame(
    pos_t, vel_t, quat_t, omega_t, motor_w_norm,
    gate_pos_t, gate_yaw_t, next_pos_rel, next_yaw_rel,
) -> np.ndarray:
    """Build the 20-D QuadRace observation for a single drone, all inputs in
    THEIR frame. `quat_t` = (qw,qx,qy,qz) body->world; `motor_w_norm` = 4 motor
    speeds already normalized to [-1,1]; `next_pos_rel`/`next_yaw_rel` are the
    target+1 gate's entries from `compute_relative_gates` (zeros if none)."""
    pos_t = np.asarray(pos_t, dtype=np.float64)
    vel_t = np.asarray(vel_t, dtype=np.float64)
    gate_pos_t = np.asarray(gate_pos_t, dtype=np.float64)
    c, s = np.cos(gate_yaw_t), np.sin(gate_yaw_t)

    dx, dy = pos_t[0] - gate_pos_t[0], pos_t[1] - gate_pos_t[1]
    pos_g = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)
    vel_g = np.array([c * vel_t[0] + s * vel_t[1],
                      -s * vel_t[0] + c * vel_t[1]], dtype=np.float64)

    phi, theta, psi = their_euler_from_quat(quat_t)
    psi = _wrap_pi(psi - gate_yaw_t)
    phi, theta = _wrap_pi(phi), _wrap_pi(theta)

    obs = np.zeros(20, dtype=np.float32)
    obs[0:2] = pos_g
    obs[2] = pos_t[2] - gate_pos_t[2]
    obs[3:5] = vel_g
    obs[5] = vel_t[2]
    obs[6], obs[7], obs[8] = phi, theta, psi
    obs[9:12] = np.asarray(omega_t, dtype=np.float64)
    obs[12:16] = np.asarray(motor_w_norm, dtype=np.float64)
    obs[16:19] = np.asarray(next_pos_rel, dtype=np.float64)
    obs[19] = float(next_yaw_rel)
    return obs


def motor_speed_to_norm(motor_w_phys) -> np.ndarray:
    """Physical motor angular velocity (rad/s) -> their normalized w in [-1,1]."""
    return 2.0 * np.asarray(motor_w_phys, dtype=np.float64) / W_MAX_N - 1.0


# --------------------------------------------------------------------------- #
# Their motor command curve (action U in [-1,1] -> steady-state rpm Wc)
# --------------------------------------------------------------------------- #
def action_to_motor_speed(u, w_min=341.75, w_max=3100.0, k=0.5) -> np.ndarray:
    """Their steady-state motor response (environment.py:69-80):
        U = (u+1)/2;  Wc = (w_max-w_min)*sqrt(k*U^2 + (1-k)*U) + w_min
    Maps the policy's per-motor command to a rad/s reference for our sim."""
    u = np.asarray(u, dtype=np.float64)
    cap = (u + 1.0) / 2.0
    return (w_max - w_min) * np.sqrt(np.clip(k * cap * cap + (1.0 - k) * cap,
                                             0.0, None)) + w_min
