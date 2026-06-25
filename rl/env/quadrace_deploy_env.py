"""Deploy a TU-Delft MonoRace PPO policy in our Aerostack2 sim.

`QuadRaceDeployEnv` subclasses `DroneGoalEnv` and overrides just the I/O so a
policy trained in their JAX sim (`quad_race.environment.QuadRace`) drives our
multirotor sim directly:

  * observation  -> their 20-D gate-relative obs, rebuilt from the live drone
    via the validated `quadrace_adapter` (global Rx(180) ENU/FLU -> NED/FRD).
  * action       -> their 4 per-motor commands U in [-1,1], mapped through their
    steady-state rpm curve `Wc(U)` and published as MOTOR_W motor speeds.

Engage model (per user): NO takeoff. Arm + offboard + ACRO, and the policy
flies UP from the floor itself — matching their `initialize_on_ground=True`
training. The track is held in ENU (our frame); only the obs is transformed.

This reuses DroneGoalEnv's motor plumbing (`_motors_pub`, `motor_speed` sub,
`_read_*`, `_teleport`, `_set_physics`). The config MUST set `action.mode:
motor` so that plumbing is built.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .drone_goal_env import DroneGoalEnv
from . import quadrace_adapter as qa

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    from gym import spaces


class QuadRaceDeployEnv(DroneGoalEnv):
    def __init__(self, config_path, drone_namespace=None, *,
                 gates_enu, start_dist=2.0, ground_z=0.05,
                 w_min=341.75, w_max=3100.0, k=0.5,
                 motor_perm=(0, 1, 2, 3), step_sleep=None, free_run=True):
        if not gates_enu:
            raise ValueError("gates_enu must be a non-empty list of gate dicts")
        self._gates_enu = [dict(g) for g in gates_enu]
        self._start_dist = float(start_dist)
        self._ground_z = float(ground_z)
        self._step_sleep_arg = step_sleep
        # free_run: real-time control loop, physics NEVER paused, motor commands
        # streamed paced to dt (~100 Hz). Avoids the 2 pause/resume service
        # round-trips per step that cap the paced path at ~58 Hz.
        self._free_run = bool(free_run)
        self._fr_deadline = None
        self._wmin, self._wmax, self._k = float(w_min), float(w_max), float(k)
        # motor_perm[i] = OUR sim motor index that plays the role of THEIR motor i
        self._motor_perm = np.asarray(motor_perm, dtype=int)
        self._inv_perm = np.argsort(self._motor_perm)

        super().__init__(config_path, drone_namespace, publish_gate_marker=False)
        if self._action_mode != 'motor':
            raise RuntimeError(
                "QuadRaceDeployEnv requires action.mode: motor in the config "
                f"(got {self._action_mode!r}).")

        # Their policy I/O (QuadRace defaults: R6_input=False, gates_ahead=1,
        # no history/param => obs 20, action 4 in [-1,1]).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Their-frame gate tables (+ looped relative-gate lookahead table).
        gp = np.array([qa.vec_enu_to_their([g['x'], g['y'], g['z']])
                       for g in self._gates_enu], dtype=np.float64)
        gy = np.array([qa.yaw_enu_to_their(g['yaw'])
                       for g in self._gates_enu], dtype=np.float64)
        self._gp_their, self._gy_their = gp, gy
        self._gpr, self._gyr = qa.compute_relative_gates(gp, gy)
        self._n_gates = len(self._gates_enu)
        # Sim-dt calibration: resuming the clock (a service-poll round-trip)
        # leaks ~one poll period (~5 ms) of sim-time BEFORE the controlled
        # sleep, so the default step integrates ~dt+0.005 = 0.015 s (measured,
        # std 0). Sleep dt-0.005 so the unpaused window lands on ~dt=0.01, the
        # rate the policy trained at. Tune with rl.scripts.check_deploy_dt.
        self._step_sleep = (float(self._step_sleep_arg)
                            if self._step_sleep_arg is not None
                            else max(0.0, self._dt - 0.005))
        self._target_gate_idx = 0
        self._set_target_gate(0)

    # ------------------------------------------------------------------ #
    # Engage: arm + offboard + ACRO, NO takeoff (policy lifts off itself)
    # ------------------------------------------------------------------ #
    def _takeoff_and_engage(self) -> None:
        from as2_msgs.msg import ControlMode
        from as2_msgs.srv import SetControlMode
        if not self.drone.arm():
            raise RuntimeError('Failed to arm the drone.')
        if not self.drone.offboard():
            raise RuntimeError('Failed to switch to offboard.')
        # Engage ACRO so the FSM accepts actuator commands; the motors topic
        # then flips the sim into MOTOR_W (same trick as motor/rates modes).
        req = SetControlMode.Request()
        req.control_mode.control_mode = ControlMode.ACRO
        req.control_mode.yaw_mode = ControlMode.YAW_SPEED
        req.control_mode.reference_frame = ControlMode.UNDEFINED_FRAME
        future = self._set_mode_client.call_async(req)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if not future.done() or future.result() is None \
                or not future.result().success:
            raise RuntimeError('set_platform_control_mode(ACRO) failed/timed out.')
        # Prime motors OFF (drone rests on the floor; the policy spins them up).
        self._send_motor_command([0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    # Track / target management
    # ------------------------------------------------------------------ #
    def _set_target_gate(self, idx: int) -> None:
        self._target_gate_idx = int(idx) % self._n_gates
        g = self._gates_enu[self._target_gate_idx]
        self._target_pos = np.array([g['x'], g['y'], g['z']], dtype=np.float64)
        self._target_yaw = float(g['yaw'])

    def set_target_gate(self, idx: int) -> None:
        """Public: point the policy at gate `idx` (no teleport)."""
        self._set_target_gate(idx)

    @property
    def target_gate_idx(self) -> int:
        return self._target_gate_idx

    # ------------------------------------------------------------------ #
    # Observation: their 20-D gate-relative obs from live readings
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        pos_enu, _yaw = self._read_pose()
        r_wb = self._read_rotation()
        omega = self._read_omega()
        motor_w = self._read_motor_speed()      # OUR sim motor order, rad/s
        # World velocity = R_wb @ body_velocity. self_localization/twist is
        # FULL body (FLU) frame (the platform rotates the sim's world velocity
        # by orientation.inverse() before publishing), so we must rotate by the
        # FULL R_wb here. The base _read_velocity() uses yaw-only, which is
        # wrong while pitched/rolled and made the policy slam after gate 1 —
        # the bindings sim feeds exact world velocity, so this matches it.
        vel_body = np.asarray(self.drone.speed, dtype=np.float64)
        vel_enu = r_wb @ vel_body

        pos_t = qa.vec_enu_to_their(pos_enu)
        vel_t = qa.vec_enu_to_their(vel_enu)
        quat_t = qa.rotmat_to_quat(qa.rotmat_enu_to_their(r_wb))
        omega_t = qa.body_rates_enu_to_their(omega)
        # present motor speeds in THEIR order, normalized to [-1,1]
        motor_w_their = motor_w[self._motor_perm]
        motor_norm = qa.motor_speed_to_norm(motor_w_their)

        idx = self._target_gate_idx
        nxt = (idx + 1) % self._n_gates
        obs = qa.assemble_obs_their_frame(
            pos_t, vel_t, quat_t, omega_t, motor_norm,
            self._gp_their[idx], self._gy_their[idx],
            self._gpr[nxt], self._gyr[nxt])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Action: their U -> Wc curve -> MOTOR_W (with motor permutation)
    # ------------------------------------------------------------------ #
    def _apply_action(self, action: np.ndarray) -> None:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._action_prev = self._last_action
        self._last_action = a.astype(np.float32)
        if not self._free_run:
            self._set_physics(False)   # paced path resumes per step; free_run leaves it running
        wc_their = qa.action_to_motor_speed(a, self._wmin, self._wmax, self._k)
        # their motor i drives our sim motor motor_perm[i]
        wc_ours = wc_their[self._inv_perm]
        self._send_motor_command(wc_ours)

    # ------------------------------------------------------------------ #
    # Step: apply (resume clock), integrate ~dt, freeze, return obs.
    # Overrides DroneGoalEnv.step to use the calibrated _step_sleep so the
    # integrated sim-dt is ~0.01 (the policy's training rate) instead of the
    # ~0.015 the default sleep(dt) produces (clock-resume leak; see __init__).
    # ------------------------------------------------------------------ #
    def step(self, action):
        if self._free_run:
            # Real-time loop: physics runs free; pace publishes to dt via an
            # absolute deadline so the rate stays ~1/dt Hz regardless of compute.
            self._apply_action(action)
            self._fr_deadline += self._dt
            slack = self._fr_deadline - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                self._fr_deadline = time.time()   # fell behind -> resync
            return self._observe_step()
        # Paced (frozen-physics) path — exact-ish dt but ~58 Hz max.
        self._apply_action(action)
        time.sleep(self._step_sleep)
        self._set_physics(True)
        return self._observe_step()

    # ------------------------------------------------------------------ #
    # Gate-pass / done are decided by the driver script (mirrors test_track).
    # ------------------------------------------------------------------ #
    def _observe_step(self):
        pos_enu, yaw = self._read_pose()
        obs = self._get_obs()
        self._step_idx += 1
        info = {
            'pos_enu': pos_enu,
            'yaw_enu': yaw,
            'target_gate_idx': self._target_gate_idx,
            'step': self._step_idx,
        }
        return obs, 0.0, False, False, info

    # ------------------------------------------------------------------ #
    # Reset: teleport to the floor behind gate 0, facing the crossing dir.
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None, start_gate=0):
        self._set_target_gate(start_gate)
        g = self._gates_enu[self._target_gate_idx]
        cos_t, sin_t = math.cos(g['yaw']), math.sin(g['yaw'])
        start_pos = np.array([
            g['x'] - self._start_dist * cos_t,
            g['y'] - self._start_dist * sin_t,
            self._ground_z,                      # on the floor
        ], dtype=np.float64)
        self._teleport(start_pos, float(g['yaw']))
        self._send_motor_command([0.0, 0.0, 0.0, 0.0])
        self._set_physics(True)                  # freeze during teleport settle
        time.sleep(3.0 * self._dt)               # let state publishers catch up

        self._step_idx = 0
        self._last_action = np.zeros(4, dtype=np.float32)
        self._action_prev = np.zeros(4, dtype=np.float32)
        if self._free_run:
            self._set_physics(False)             # resume; physics free-runs from here
            self._fr_deadline = time.time()
        return self._get_obs(), {
            'target_gate_idx': self._target_gate_idx,
            'start_pos': start_pos,
        }
