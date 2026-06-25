"""Goal-reaching Gymnasium environment for an Aerostack2 multirotor drone.

The drone is armed, set to OFFBOARD and taken off once at construction. Each
``reset`` teleports the simulator to a randomized pose using the
``set_platform_state`` ROS service exposed by the patched
``as2_platform_multirotor_simulator``. ``step`` sends earth-frame velocity
references through ``motion_ref_handler.speed``.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
import yaml
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from actuator_msgs.msg import Actuators
from as2_msgs.msg import ControlMode, Thrust
from as2_msgs.srv import SetControlMode, SetPlatformState
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped, Vector3
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))  # (w, x, y, z)


def _euler_to_quat(roll: float, pitch: float,
                   yaw: float) -> tuple[float, float, float, float]:
    """ZYX (yaw→pitch→roll) intrinsic Euler to (w, x, y, z) quaternion.

    Used by the exploring-starts teleport to spawn the drone at a random
    roll/pitch attitude (not just yaw), so the policy is forced to learn
    recovery / control from tilted states from step 0.
    """
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        cr * cp * cy + sr * sp * sy,   # w
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
    )


def _quat_to_rotmat(qw: float, qx: float, qy: float,
                    qz: float) -> np.ndarray:
    """Body→world rotation matrix R_wb from a (w, x, y, z) attitude quaternion.

    Used as the SINGLE attitude source for the observation geometry (camera
    pointing + 6D rotation). Sourcing R_wb from the quaternion — rather than
    reconstructing it from euler roll/pitch/yaw — is exact and singularity-free
    (no ±90° pitch gimbal fold). Falls back to identity for a degenerate
    (zero-norm) quaternion.
    """
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _camera_pointing(rel_world: np.ndarray, R_wb: np.ndarray,
                     camera_pitch: float) -> tuple[np.ndarray, float]:
    """Where the gate sits in the PnP camera, from the body→world rotation R_wb.

    Returns ``(los_cam, cam_off_axis)``:
      - los_cam      — the gate line-of-sight as a UNIT vector in the camera
                       basis: ``[fwd, right, up]`` (forward = optical axis,
                       right = image right, up = image up). Continuous on the
                       sphere — no atan2 wrap when the gate passes 90°/behind.
      - cam_off_axis — angle between the optical axis and the line-of-sight
                       (= acos(fwd)). ROLL-INVARIANT (only "how centered"); any
                       attitude that aims the camera at the gate scores 0.

    ``rel_world`` is the gate position relative to the drone in world ENU. The
    camera is rigidly mounted, optical axis pitched ``camera_pitch`` UP from
    body +x (FPV uptilt): f = [cos c, 0, +sin c] in FLU, image right = body −y,
    image up = [−sin c, 0, cos c]. {f, −y, u} is orthonormal, so [fwd,right,up]
    is a unit vector.
    """
    rel_world = np.asarray(rel_world, dtype=np.float64)
    n_los = float(np.linalg.norm(rel_world))
    los_b = (R_wb.T @ (rel_world / n_los)
             if n_los > 1e-9 else np.array([1.0, 0.0, 0.0]))
    c = camera_pitch
    f_cam = np.array([math.cos(c), 0.0, math.sin(c)])    # forward + up
    u_cam = np.array([-math.sin(c), 0.0, math.cos(c)])   # image up
    fwd = float(np.dot(los_b, f_cam))
    right = float(-los_b[1])                             # · [0, -1, 0]
    up = float(np.dot(los_b, u_cam))
    cam_off_axis = math.acos(max(-1.0, min(1.0, fwd)))   # roll-invariant
    return np.array([fwd, right, up]), cam_off_axis


def _rotation_6d(R_wb: np.ndarray, target_yaw: float) -> np.ndarray:
    """Continuous 6D encoding of the drone's attitude relative to the GATE FRAME
    (Zhou et al. 2019, "On the Continuity of Rotation Representations").

    Returns the first two columns of the gate→body rotation, i.e. the gate
    frame's x-axis (crossing normal) and y-axis (left) expressed in the body
    frame — 6 numbers, each in [-1, 1].

    R_gate_body = Rz(−target_yaw) · R_wb. The reference is the gate's STABLE
    per-episode orientation, so this rotation changes only when the drone
    actually rotates — never with its position (unlike a line-of-sight / bearing
    reference, which spins as the drone moves and is singular AT the gate). It is
    yaw-RELATIVE (uses target_yaw, not absolute world yaw → no heading leak) and,
    being relative to the gate, it also encodes the gate facing / which-side.
    Built from R_wb (quaternion-sourced) so it is fully continuous: no ±180° yaw
    wrap (cos/sin), roll unique through inversion, no ±90° pitch gimbal. The
    third gate axis (up) is the cross product of the two columns.
    """
    ct, st = math.cos(target_yaw), math.sin(target_yaw)
    rz_neg = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])  # Rz(−ψ)
    r_gate_body = rz_neg @ R_wb                  # body → gate
    r_body_gate = r_gate_body.T                  # gate → body (cols = gate axes in body)
    return np.concatenate([r_body_gate[:, 0], r_body_gate[:, 1]])  # 6 numbers


class DroneGoalEnv(gym.Env):
    """Drive the drone to a randomized (x, y, z, yaw) target."""

    metadata = {'render_modes': []}

    def __init__(self, config_path: str | Path,
                 drone_namespace: str | None = None,
                 publish_gate_marker: bool = True):
        super().__init__()
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        if not rclpy.ok():
            rclpy.init()

        drone_cfg = self.cfg['drone']
        if drone_namespace is None:
            namespaces = drone_cfg.get('namespaces')
            drone_namespace = (
                namespaces[0] if namespaces else drone_cfg.get('namespace', 'drone0')
            )
        self.drone_namespace = drone_namespace

        self.drone = DroneInterfaceTeleop(
            drone_id=self.drone_namespace,
            verbose=False,
            use_sim_time=True,
            spin_rate=float(drone_cfg['spin_rate_hz']),
        )

        # A separate node + executor handles the teleport service so it
        # doesn't share the DroneInterface's spin loop.
        self._svc_node = Node(f'rl_env_svc_{self.drone_namespace}')
        self._svc_executor = SingleThreadedExecutor()
        self._svc_executor.add_node(self._svc_node)
        self._svc_thread = threading.Thread(
            target=self._svc_executor.spin, daemon=True)
        self._svc_thread.start()

        # Cache the latest attitude quaternion (w, x, y, z) straight off
        # self_localization/pose — the SAME PoseStamped DroneInterface consumes,
        # but keeping the quaternion (DroneInterface throws it away as euler).
        # Used to build a singularity-free body→world rotation for the
        # observation geometry (camera pointing + 6D attitude). The
        # _svc_executor thread writes; step()/reset() read (atomic tuple
        # rebind, no lock needed). None until the first pose arrives → R_wb
        # falls back to euler (see _read_rotation). Match the publisher's
        # sensor-data QoS or messages are dropped on the floor.
        self._latest_quat: tuple[float, float, float, float] | None = None
        self._pose_quat_sub = self._svc_node.create_subscription(
            PoseStamped, f'/{self.drone_namespace}/self_localization/pose',
            self._pose_quat_callback, qos_profile_sensor_data)

        # Cache the latest BODY-FRAME angular velocity (measured body rates ω)
        # off self_localization/twist — DroneInterface consumes the same topic
        # but keeps only the linear part. ω is part of the rotational state and
        # the body-rate reward penalty operates on it. None until the first
        # message → falls back to zeros (see _read_omega).
        self._latest_omega: tuple[float, float, float] | None = None
        self._twist_omega_sub = self._svc_node.create_subscription(
            TwistStamped, f'/{self.drone_namespace}/self_localization/twist',
            self._twist_omega_callback, qos_profile_sensor_data)

        self._set_state_client = self._svc_node.create_client(
            SetPlatformState,
            f"/{self.drone_namespace}/set_platform_state",
        )
        if not self._set_state_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                'set_platform_state service is unavailable after 10s. Make '
                'sure the patched as2_platform_multirotor_simulator is running.'
            )

        # Pause client: freezes this drone's physics after reset() so it
        # doesn't drift while the policy computes the next action. The
        # service lives on the per-drone sim_clock_publisher node — one
        # /clock authority per DDS domain. Pausing halts the /clock
        # publisher, which freezes every sim-time-bound timer in this
        # domain (platform integrator, controller, state estimator).
        self._pause_client = self._svc_node.create_client(
            SetBool, '/sim_clock_publisher/pause_physics')
        if not self._pause_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                '/sim_clock_publisher/pause_physics is unavailable after 10s. '
                'Make sure sim_clock_publisher_node is running on this drone\'s '
                'ROS_DOMAIN_ID (it is launched per-drone by tmuxinator/'
                'aerostack2.yaml in the "platform" window).'
            )
        self._physics_paused = False

        # Action interpretation — see the `action.mode` block in
        # rl_config.yaml for the two layouts:
        #   "speed" → [vx, vy, vz, yaw_rate]    (default; routed through the
        #             AS2 motion controller via DroneInterface.motion_ref)
        #   "rates" → [roll_rate, pitch_rate, yaw_rate, thrust] (publishes
        #             straight to /<ns>/actuator_command/{thrust,twist},
        #             engaging ControlMode.ACRO on the platform).
        action_cfg = self.cfg.get('action', {})
        self._action_mode = str(action_cfg.get('mode', 'speed')).lower()
        if self._action_mode not in ('speed', 'rates', 'motor'):
            raise ValueError(
                f"action.mode must be 'speed', 'rates' or 'motor', got "
                f"{self._action_mode!r}.")
        # rates and motor are both low-level modes: they engage ACRO on the
        # platform (to satisfy the AS2 FSM / accept actuator commands) and both
        # expose body rates ω in the observation.
        self._motors_pub = None
        self._latest_motor_speed: tuple[float, float, float, float] | None = None
        if self._action_mode == 'rates':
            rates_cfg = action_cfg.get('rates') or {}
            self._roll_rate_max = float(rates_cfg.get('roll_rate_max', 22.0))
            self._pitch_rate_max = float(rates_cfg.get('pitch_rate_max', 22.0))
            self._yaw_rate_max_rates = float(rates_cfg.get(
                'yaw_rate_max', 22.0))
            # tanh scale for the measured-body-rate (ω) obs channels: the
            # largest commanded rate limit, so the operating range maps to
            # ≈ ±0.76 with graceful saturation for overshoot.
            self._w_obs_scale = max(
                self._roll_rate_max, self._pitch_rate_max,
                self._yaw_rate_max_rates, 1e-3)
            self._thrust_min = float(rates_cfg.get('thrust_min', 0.0))
            self._thrust_max = float(rates_cfg.get('thrust_max', 30.0))
            # Hover thrust = vehicle mass × g; sets the zero-action thrust
            # for the hover-centered mapping used by _apply_action. Cached
            # on the env so __init__ / reset() prime publishes and the
            # action mapping all stay in sync if the config changes.
            self._thrust_hover = float(rates_cfg.get('thrust_hover', 9.81))
            if not (self._thrust_min <= self._thrust_hover <= self._thrust_max):
                raise ValueError(
                    f'action.rates.thrust_hover ({self._thrust_hover}) '
                    f'must lie within [thrust_min={self._thrust_min}, '
                    f'thrust_max={self._thrust_max}].')
            # Direct actuator-command publishers — bypass the motion
            # controller because the PID controller in this stack doesn't
            # accept ACRO as input. The platform subscribes to these topics
            # on the same DDS domain.
            self._thrust_pub = self._svc_node.create_publisher(
                Thrust, f'/{self.drone_namespace}/actuator_command/thrust', 10)
            self._twist_pub = self._svc_node.create_publisher(
                TwistStamped, f'/{self.drone_namespace}/actuator_command/twist',
                10)
            # Service to put the platform into ACRO. Path matches the AS2
            # convention used by mock_aerial_platform and the docs.
            self._set_mode_client = self._svc_node.create_client(
                SetControlMode,
                f'/{self.drone_namespace}/set_platform_control_mode')
            if not self._set_mode_client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    f'/{self.drone_namespace}/set_platform_control_mode '
                    'unavailable after 10s. Rates mode needs the platform '
                    'to expose this service.')
        elif self._action_mode == 'motor':
            # Direct per-motor control (winning A2RL formulation): the policy
            # commands the 4 motor angular velocities — no inner attitude/rate
            # controller (the policy IS the controller). The patched platform
            # routes actuator_command/motors → simulator MOTOR_W and publishes
            # the actual (lagged) motor speeds on motor_speed for the obs.
            motor_cfg = action_cfg.get('motor') or {}
            self._motor_max_speed = float(motor_cfg.get('max_speed', 1139.34272))
            self._motor_min_speed = float(motor_cfg.get('min_speed', 0.0))
            self._motor_limit = float(motor_cfg.get('motor_limit', 1.0))
            self._motor_u_lim = 2.0 * self._motor_limit - 1.0   # action upper bound
            self._motor_hover_speed = float(motor_cfg.get('hover_speed', 510.0))
            # ω-obs scale (motor mode has no rate limit) and motor-rpm-obs scale.
            self._w_obs_scale = float(motor_cfg.get('w_obs_scale', 20.0))
            self._motor_obs_scale = max(self._motor_max_speed, 1e-6)
            self._thrust_pub = None
            self._twist_pub = None
            self._motors_pub = self._svc_node.create_publisher(
                Actuators, f'/{self.drone_namespace}/actuator_command/motors', 10)
            # Cache the actual motor speeds for the observation (mirrors the
            # pose-quaternion / twist-ω subs). None until the first message.
            self._motor_speed_sub = self._svc_node.create_subscription(
                Actuators, f'/{self.drone_namespace}/motor_speed',
                self._motor_speed_callback, qos_profile_sensor_data)
            # Engage ACRO so the AS2 FSM accepts actuator commands; the motors
            # topic then overrides the sim into MOTOR_W (same FSM trick as rates).
            self._set_mode_client = self._svc_node.create_client(
                SetControlMode,
                f'/{self.drone_namespace}/set_platform_control_mode')
            if not self._set_mode_client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    f'/{self.drone_namespace}/set_platform_control_mode '
                    'unavailable after 10s. Motor mode needs the patched '
                    'platform exposing this service + actuator_command/motors.')
        else:
            self._thrust_pub = None
            self._twist_pub = None
            self._set_mode_client = None

        # Observation (shared 9-D core: body-frame relative position to target
        # (3) + body-frame drone velocity (3) + camera LOS unit vector
        # [fwd,right,up] (3)), then mode-specific. ALL channels are continuous /
        # singularity-free:
        #   speed mode (10-D): core + relative yaw / π (1). The motion controller
        #     handles attitude, so only relative heading matters.
        #   rates mode (22-D): core + 6D rotation (6) + measured body rates ω (3)
        #     + previous action (4). Follows the Environment-as-Policy (UZH RPG,
        #     ICRA 2025) in-rates state [R̃, v, ω, a_prev, gate], with the gate
        #     encoded NOT as corners: the 6D (first two columns of the gate→body
        #     rotation, Zhou et al. 2019) is the drone's attitude relative to the
        #     stable GATE FRAME, so it carries gate orientation / which-side, and
        #     the gate position is the relative-position core. Singularity-free
        #     (quaternion-sourced): no ±180° yaw wrap, no ±90° pitch gimbal, roll
        #     unique through inversion. ω is the rotational state; a_prev pairs
        #     with the action-smoothness reward. The camera uvec replaces the old
        #     atan2 azimuth/elevation. See `_rotation_6d` / `_camera_pointing`.
        #   motor mode (26-D): the rates-mode 22-D + 4 actual motor speeds
        #     (rad/s, normalized). Direct per-motor control (winning A2RL setup):
        #     no inner controller, and the observable motor speeds make the
        #     actuator-lag (tau) POMDP Markov.
        self._obs_dim = {'speed': 10, 'rates': 22, 'motor': 26}[
            self._action_mode]
        # Camera mount tilt (forward/UP from body +x; FPV uptilt), radians.
        # Drives the full-3D camera pointing: the camera-frame azimuth/
        # elevation obs channels and the roll-invariant off-axis pointing
        # penalty, so "look at the gate" accounts for the whole attitude
        # (roll included), not just yaw azimuth + pitch elevation.
        self._camera_pitch = math.radians(
            float(self.cfg.get('camera', {}).get('pitch_deg', 30.0)))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

        self._dt = 1.0 / float(self.cfg['control_hz'])
        self._max_steps = int(self.cfg['episode']['max_steps'])
        self._step_idx = 0
        # Initialize target from config so the gate marker can publish before
        # the first reset(); reset() re-samples this (currently a no-op for
        # the fixed config target).
        self._target_pos = np.array([
            float(self.cfg['target']['x']),
            float(self.cfg['target']['y']),
            float(self.cfg['target']['z']),
        ], dtype=np.float64)
        # Fixed target yaw — the drone's preferred final orientation. Used
        # for the relative-yaw observation, continuous alignment penalty, and
        # the terminal alignment bonus.
        self._target_yaw = float(self.cfg['target'].get('yaw', 0.0))
        # Action history for the previous-action obs channel and the
        # action-smoothness reward: _last_action = u_t (the action applied this
        # step), _action_prev = u_{t-1}. _apply_action rolls them forward.
        self._last_action = np.zeros(4, dtype=np.float32)
        self._action_prev = np.zeros(4, dtype=np.float32)
        self._closed = False
        # Spawn-to-target distance recorded at every reset(); drives the
        # terminal time-budget for the success bonus. Initialized to 0
        # here so the first reset doesn't read an undefined attribute if
        # something goes wrong; the real value is set in reset() before
        # the first _observe_step would consume it.
        self._initial_dist_to_target = 0.0
        # Previous-step 3D distance to target, for the dense progress
        # reward. Seeded to the spawn distance in reset() before the first
        # _observe_step consumes it.
        self._prev_dist_to_target = 0.0
        # Success-arming latch. A gate-crossing success can only fire after
        # the drone has been observed clearly OUTSIDE the gate plane since
        # the last reset. Without this, the first post-reset observation —
        # which can still read the previous episode's at-the-gate pose
        # before the teleport propagates through the pose pipeline (the
        # gate is pinned at the same world point every episode) — fires a
        # phantom success at step 0. Reset to False in reset().
        self._success_armed = False

        # Single source of truth for both the observation normalization
        # divisors AND the OOB box: symmetric half-extents around the
        # *current* target. The policy sees `rel / extent`, saturating at
        # ±1 exactly when the env would call OOB — no mismatch between
        # what the network reads and what the env penalises.
        #
        # Preferred config:
        #   oob_box:
        #     x: 10.0
        #     y: 10.0
        #     z:  3.0
        #
        # Fallback if `oob_box` is absent: derive symmetric extents from the
        # `workspace` block (max of the two sides) so older configs still work.
        # ws = self.cfg['workspace']
        tx = float(self.cfg['target']['x'])
        ty = float(self.cfg['target']['y'])
        tz = float(self.cfg['target']['z'])
        oob_cfg = self.cfg.get('oob_box')
        if oob_cfg is not None:
            self._oob_x_ext = float(oob_cfg['x'])
            self._oob_y_ext = float(oob_cfg['y'])
            self._oob_z_ext = float(oob_cfg['z'])
        # else:
        #     self._oob_x_ext = max(
        #         abs(float(ws['x_max']) - tx), abs(float(ws['x_min']) - tx))
        #     self._oob_y_ext = max(
        #         abs(float(ws['y_max']) - ty), abs(float(ws['y_min']) - ty))
        #     self._oob_z_ext = max(
        #         abs(float(ws['z_max']) - tz), abs(float(ws['z_min']) - tz))
        # Normalization divisors used by _get_obs / _compute_reward. Aliased
        # to the OOB extents so obs[i] saturates at ±1 exactly at OOB.
        self._pos_max_x = self._oob_x_ext
        self._pos_max_y = self._oob_y_ext
        self._pos_max_z = self._oob_z_ext
        self._max_dist_xy = math.sqrt(self._pos_max_x ** 2 + self._pos_max_y ** 2)

        self._takeoff_and_engage()

        # Every drone publishes its OWN gate marker on /rl/target_gate. Each
        # uses a unique Marker.id derived from the namespace so RViz tracks
        # them independently — N drones with N randomized targets show up as
        # N gates simultaneously.
        self._gate_marker_pub = None
        self._gate_marker_timer = None
        # Parse a numeric id from the namespace ("drone0" → 0, "drone17" → 17)
        # for the Marker.id; falls back to 0 if no digits present.
        digits = ''.join(c for c in self.drone_namespace if c.isdigit())
        self._gate_marker_id = int(digits) if digits else 0
        if publish_gate_marker:
            self._gate_marker_pub = self._svc_node.create_publisher(
                Marker, '/rl/target_gate', 10)
            self._gate_marker_timer = self._svc_node.create_timer(
                1.0, self._publish_gate_marker)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _takeoff_and_engage(self) -> None:
        """Arm, offboard, take off, then pin the platform in the control
        mode that matches `action.mode`.

        Takeoff always goes through AS2's takeoff behavior (which engages
        SPEED internally). For ``action.mode: rates`` we then explicitly
        switch the platform to ACRO + YAW_SPEED via
        ``set_platform_control_mode`` and prime with one hover-thrust /
        zero-rates command so the platform has a valid reference before
        the first env step.
        """
        if not self.drone.arm():
            raise RuntimeError('Failed to arm the drone.')
        if not self.drone.offboard():
            raise RuntimeError('Failed to switch to offboard.')
        if not self.drone.takeoff(
                height=float(self.cfg['takeoff_height']),
                speed=float(self.cfg['takeoff_speed'])):
            raise RuntimeError('Takeoff failed.')

        if self._action_mode == 'speed':
            # Sending a zero-velocity SPEED command pins the platform in
            # SPEED + YAW_SPEED control mode for the rest of training.
            self._send_speed_command(0.0, 0.0, 0.0, 0.0)
            return

        # action.mode == 'rates' → flip the platform to ACRO + YAW_SPEED.
        req = SetControlMode.Request()
        req.control_mode.control_mode = ControlMode.ACRO
        req.control_mode.yaw_mode = ControlMode.YAW_SPEED
        req.control_mode.reference_frame = ControlMode.UNDEFINED_FRAME
        future = self._set_mode_client.call_async(req)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(
                'set_platform_control_mode(ACRO + YAW_SPEED) timed out.')
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(
                'Platform refused ACRO control mode. Check that ACRO is '
                'enabled in the platform\'s control_modes.yaml.')

        # Prime the platform with a hover command so it has a reference to
        # track from t=0. Rates: hover thrust + zero rates. Motor: hover motor
        # speed on all 4 (this also flips the sim into MOTOR_W).
        if self._action_mode == 'motor':
            self._send_motor_command([self._motor_hover_speed] * 4)
        else:
            self._send_rates_command(0.0, 0.0, 0.0, self._thrust_hover)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Make sure the simulator isn't left paused when we tear down.
        try:
            self._set_physics(False)
        except Exception:
            pass
        try:
            self._svc_executor.shutdown()
        except Exception:
            pass
        try:
            self._svc_node.destroy_node()
        except Exception:
            pass
        try:
            self.drone.shutdown()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.try_shutdown()

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self._target_pos, self._target_yaw = self._sample_target_pose()

        # Push the new target to RViz immediately so the gate marker doesn't
        # lag the actual target by up to the 1 Hz timer period (which is what
        # made it look like the drone was "hovering at someone else's gate").
        if self._gate_marker_pub is not None:
            self._publish_gate_marker()

        min_dist = float(self.cfg.get('min_init_target_dist', 0.0))
        for _ in range(50):
            init_pos, init_yaw = self._sample_init_pose()
            if np.linalg.norm(init_pos - self._target_pos) >= min_dist:
                break

        # Record the straight-line spawn-to-target distance BEFORE teleport
        # so the time-budget computation in _observe_step uses the actual
        # initial spawn pose, not whatever the drone drifted to mid-episode.
        # Used by the terminal time_factor in the success bonus, and as the
        # seed for the dense progress reward's previous-distance tracker.
        self._initial_dist_to_target = float(
            np.linalg.norm(init_pos - self._target_pos))
        self._prev_dist_to_target = self._initial_dist_to_target
        # Exploring starts: spawn already moving / tilted so the policy must
        # learn aggressive-regime control from step 0.
        es_roll, es_pitch, es_lin_vel, es_ang_vel = (
            self._sample_exploring_start(init_pos))
        self._teleport(init_pos, init_yaw, roll=es_roll, pitch=es_pitch,
                       lin_vel=es_lin_vel, ang_vel=es_ang_vel)

        # Re-issue a zero-effort hold so the platform's "last reference"
        # tracks the just-teleported pose. Mode-specific:
        #   speed → 0 velocity + 0 yaw_rate (also re-pins SPEED mode if the
        #           motion controller had drifted off it).
        #   rates → hover thrust + 0 angular rates. Sending the speed
        #           equivalent here would route through motion_ref_handler
        #           and try to renegotiate SPEED on the platform that's
        #           currently in ACRO, which fights both the env's prime
        #           and any subsequent _apply_action.
        if self._action_mode == 'speed':
            self._send_speed_command(0.0, 0.0, 0.0, 0.0)
        elif self._action_mode == 'motor':
            self._send_motor_command([self._motor_hover_speed] * 4)
        else:
            self._send_rates_command(0.0, 0.0, 0.0, self._thrust_hover)

        # Freeze physics immediately so the drone doesn't drift during the
        # state-publish settle sleep (or while waiting for the next action).
        # State publishers run on independent timers, so the DroneInterface
        # still receives the post-teleport pose before we read it.
        self._set_physics(True)
        time.sleep(2.0 * self._dt)

        self._step_idx = 0
        # Clear action history so the first obs reports "no prior command" and
        # the first action-smoothness penalty is measured against zeros.
        self._last_action = np.zeros(4, dtype=np.float32)
        self._action_prev = np.zeros(4, dtype=np.float32)
        # Disarm success until the drone is seen outside the gate plane.
        # _check_done arms it once abs(gate_depth) >= depth_tol, so a stale
        # first observation that still reads the drone at the gate cannot
        # register a phantom crossing.
        self._success_armed = False
        return self._get_obs(), {
            'target_pos': self._target_pos.copy(),
            'target_yaw': self._target_yaw,
            'init_pos': init_pos,
            'init_yaw': init_yaw,
        }

    def step(self, action: np.ndarray):
        # Single-env path: unpause physics + send command, sleep one control
        # period of *real* time so physics integrates exactly dt, freeze
        # physics, then read the post-action state. Vectorized swarm code
        # reuses _apply_action / _observe_step directly so the dt sleep
        # happens once per VecEnv.step instead of per-env.
        self._apply_action(action)
        time.sleep(self._dt)
        self._set_physics(True)
        return self._observe_step()

    def _apply_action(self, action: np.ndarray) -> None:
        """Send the command for ``action``; do NOT sleep or observe.

        The 4-D action is interpreted by ``self._action_mode``:
          - ``speed``: [vx, vy, vz, yaw_rate], all in [-1, +1], scaled by
            ``action.v_max`` / ``action.yaw_rate_max``.
          - ``rates``: [roll_rate, pitch_rate, yaw_rate, thrust], all in
            [-1, +1]. Rates scale by their per-axis ``*_max``; thrust uses
            a HOVER-CENTERED piecewise-linear mapping:
                a[3] = -1 → thrust_min   (max descent)
                a[3] =  0 → thrust_hover (drone hangs in place)
                a[3] = +1 → thrust_max   (max climb)
            This makes the freshly-initialized policy's neutral output
            correspond to hover instead of (thrust_min+thrust_max)/2 ≈ 15 N
            — without it a brand-new policy climbs at +5 m/s² and the
            rotational channels never get useful exploration coverage
            until PPO learns to cancel the structural climb bias.
        """
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Roll the action history forward: previous u_t becomes u_{t-1}, the
        # new command becomes u_t. Used by the previous-action obs channel and
        # the action-smoothness reward ‖u_t − u_{t-1}‖.
        self._action_prev = self._last_action
        self._last_action = a
        # Resume physics (no-op if not paused) before pushing the new command.
        self._set_physics(False)
        if self._action_mode == 'speed':
            v_max = float(self.cfg['action']['v_max'])
            yaw_rate_max = float(self.cfg['action']['yaw_rate_max'])
            vx, vy, vz = (a[:3] * v_max).tolist()
            vyaw = float(a[3] * yaw_rate_max)
            self._send_speed_command(vx, vy, vz, vyaw)
        elif self._action_mode == 'rates':
            roll_rate = float(a[0]) * self._roll_rate_max
            pitch_rate = float(a[1]) * self._pitch_rate_max
            yaw_rate = float(a[2]) * self._yaw_rate_max_rates
            a3 = float(a[3])
            if a3 >= 0.0:
                thrust = self._thrust_hover + a3 * (
                    self._thrust_max - self._thrust_hover)
            else:
                thrust = self._thrust_hover + a3 * (
                    self._thrust_hover - self._thrust_min)
            self._send_rates_command(roll_rate, pitch_rate, yaw_rate, thrust)
        else:  # 'motor' — linear map (winner's): a∈[-1,u_lim] → (a+1)/2·max_speed
            cmd = np.clip(a, -1.0, self._motor_u_lim)
            motor_speeds = (self._motor_min_speed
                            + 0.5 * (cmd + 1.0)
                            * (self._motor_max_speed - self._motor_min_speed))
            self._send_motor_command(motor_speeds)

    def _observe_step(self):
        """Compute observation, reward, and termination from a single state
        snapshot. ``_compute_state`` handles the world→body rotation once
        and we reuse every intermediate it produces — there is no second
        pose read or recomputed body-frame transform.
        """
        s = self._compute_state()
        obs = s.obs
        pos = s.pos
        vel_world = s.vel_world
        rel_world = s.rel_world
        rel_body_x = s.rel_body_x
        rel_body_y = s.rel_body_y
        vel_body_x = s.vel_body_x
        vel_body_y = s.vel_body_y
        vel_body_z = s.vel_body_z
        v_max = s.v_max
        yaw_err = s.yaw_rel  # already wrapped to [-π, π]

        dist_xy = float(math.sqrt(rel_body_x ** 2 + rel_body_y ** 2))
        bearing = s.bearing
        dist_xy_norm = dist_xy / self._max_dist_xy
        z_err_norm = abs(float(rel_world[2])) / self._pos_max_z
        bearing_norm = abs(bearing) / math.pi
        yaw_err_norm = abs(yaw_err) / math.pi
        # "Look at the gate" pointing error: the roll-INVARIANT off-axis angle
        # between the PnP camera optical axis and the HORIZONTAL line-of-sight
        # to the gate (cam_off_axis_reward, vertical LOS zeroed in
        # _compute_state). Using the horizontal LOS makes this an ATTITUDE-only
        # term — the drone must PITCH/roll/yaw to aim, and cannot satisfy it by
        # descending below the gate. Any roll/pitch/yaw that swings the camera
        # onto the gate's bearing scores well (a banked attitude is as good as a
        # level one). Normalized by π (aimed → 0, 90° off → 0.5, behind → 1.0).
        pointing_norm = min(s.cam_off_axis_reward / math.pi, 1.0)
        # 3D distance to the gate center — reused by the dense progress reward.
        curr_dist = float(np.linalg.norm(rel_world))
        # "Look at the gate" pointing penalty applies at FULL strength for the
        # whole episode (no distance fade): the policy must keep the gate
        # centered the entire approach and through the crossing.
        pointing_norm_weighted = pointing_norm
        # Forward-speed score: proportional and signed — only positive forward
        # body velocity is rewarded. Reversing through the gate (vx_body < 0)
        # gives 0 regardless of |v|, blocking the "back into the goal" exploit.
        # When yaw is aligned with target_yaw, +vx_body = +target_yaw world dir
        # = the legit gate-crossing direction.
        forward_speed_score = max(0.0, min(vel_body_x / v_max, 1.0))
        vertical_speed_norm = min(abs(vel_body_z) / v_max, 1.0)

        # Gate-local coordinates of the drone relative to the gate center.
        # Gate normal direction (along which the drone flies through) is the
        # +x axis of the gate frame, rotated by target_yaw in world ENU.
        cos_t = math.cos(self._target_yaw)
        sin_t = math.sin(self._target_yaw)
        gate_depth = cos_t * rel_world[0] + sin_t * rel_world[1]
        # In-plane: lateral (perpendicular to gate normal, horizontal) and
        # vertical (= world z, same as the PnP code's gate-local y).
        gate_lateral = -sin_t * rel_world[0] + cos_t * rel_world[1]
        gate_vertical = float(rel_world[2])
        # Velocity projected onto the gate normal in world ENU. Positive when
        # the drone is crossing in the legitimate (+target_yaw) direction;
        # negative or zero when entering from the wrong side. Used both as
        # the back-entry guard in `_check_done` and as the boolean gate on
        # `omni_speed_score` below — a wrong-side approach earns 0 speed
        # bonus no matter how fast the drone is moving overall.
        gate_normal_dot_vel = cos_t * vel_world[0] + sin_t * vel_world[1]
        crossing_speed_score = max(
            0.0, min(gate_normal_dot_vel / v_max, 1.0))
        # Aggressive terminal-speed score: take the FULL world-frame
        # velocity magnitude (all three axes) so a fast diving or laterally
        # tilted entry scores as well as a head-on horizontal one. Still
        # gated by gate_normal_dot_vel > 0 so back entries → 0. Capped at
        # v_max for reward scale stability (a drone moving at v_max in
        # multiple axes can have |v| > v_max, but we don't reward beyond
        # that — the magnitude is already 1.0 in the "full credit" sense).
        if gate_normal_dot_vel > 0.0:
            speed_magnitude = float(np.linalg.norm(vel_world))
            omni_speed_score = min(speed_magnitude / v_max, 1.0)
        else:
            speed_magnitude = 0.0
            omni_speed_score = 0.0

        reward = self._compute_reward(
            dist_xy_norm, z_err_norm, pointing_norm_weighted, yaw_err_norm,
            forward_speed_score, vertical_speed_norm)

        # Dense progress reward: + k_progress · (prev_dist − curr_dist) in
        # meters (3D). Rewards closing distance to the gate every step, so the
        # policy gets a strong dense gradient toward the gate instead of relying
        # only on the sparse terminal bonus. The sum telescopes to
        # k_progress·(initial_dist − final_dist), so it's potential-based
        # (doesn't bias WHICH path) — the speed incentive comes from the
        # discount (γ<1 front-loads progress) plus the uncapped terminal
        # time_factor.
        k_progress = float(self.cfg['reward'].get('k_progress', 0.0))
        progress_reward = k_progress * (self._prev_dist_to_target - curr_dist)
        reward += progress_reward
        self._prev_dist_to_target = curr_dist

        # Environment-as-Policy regularizers (paired with the a_prev / ω obs):
        #   r_act = − k_act · ‖u_t − u_{t-1}‖   action smoothness (damps the
        #           "noisy/oscillating command" failure mode and stops the
        #           previous-action channel from becoming a copy-the-last-action
        #           attractor — that pairing is the point);
        #   r_br  = − k_br  · ‖ω‖               body-rate penalty (discourages
        #           needless spinning / aggressive tumbling). Both use L2 norms.
        rw = self.cfg['reward']
        k_act = float(rw.get('k_act', 0.0))
        k_br = float(rw.get('k_br', 0.0))
        action_smooth = float(np.linalg.norm(
            np.asarray(self._last_action, dtype=np.float64)
            - np.asarray(self._action_prev, dtype=np.float64)))
        body_rate_mag = float(np.linalg.norm(self._read_omega()))
        smooth_reward = -k_act * action_smooth
        body_rate_reward = -k_br * body_rate_mag
        reward += smooth_reward + body_rate_reward

        terminated, truncated, info = self._check_done(
            pos, gate_depth, gate_lateral, gate_vertical,
            gate_normal_dot_vel)
        info.update({
            'dist_xy': dist_xy,
            'dist_3d': curr_dist,
            'progress_reward': progress_reward,
            'smooth_reward': smooth_reward,
            'body_rate_reward': body_rate_reward,
            'bearing': bearing,
            'yaw_err': yaw_err,
            'vel_body_x': vel_body_x,
            'vel_body_y': vel_body_y,
            'vel_body_z': vel_body_z,
            'dist_xy_norm': dist_xy_norm,
            'z_err_norm': z_err_norm,
            'bearing_norm': bearing_norm,
            'cam_azimuth': s.cam_azimuth,
            'cam_elevation': s.cam_elevation,
            'cam_off_axis': s.cam_off_axis,
            'pointing_norm': pointing_norm,
            'yaw_err_norm': yaw_err_norm,
            'forward_speed_score': forward_speed_score,
            'vertical_speed_norm': vertical_speed_norm,
            'crossing_speed_score': crossing_speed_score,
            'omni_speed_score': omni_speed_score,
            'speed_magnitude': speed_magnitude,
            'gate_depth': gate_depth,
            'gate_lateral': gate_lateral,
            'gate_vertical': gate_vertical,
        })
        if info.get('success'):
            # Gate-crossing terminal: rewards reaching the inner-opening
            # center *quickly* from spawn, regardless of crossing speed or
            # body orientation. Three factors:
            #
            #   center_factor — same as before: how close to the exact
            #     center of the inner opening the crossing was. Linear
            #     from 1 at center to 0 at the inner edge.
            #
            #   time_factor — UNBOUNDED-above speed ratio: the policy is
            #     rewarded in proportion to how fast it actually crossed,
            #     with no upper saturation, so "go faster" always pays
            #     more. `ref_speed` is just the unit reference at which the
            #     factor equals 1.0 — NOT a cap:
            #         ref_steps  = max(time_target_min,
            #                          init_dist / ref_speed · control_hz)
            #         time_factor = max(time_factor_min,
            #                           ref_steps / steps_taken)
            #     Since ref_steps / steps_taken == avg_speed / ref_speed,
            #     the factor literally equals "how many times ref_speed you
            #     averaged": cross at ref_speed → 1.0, at 2× ref_speed →
            #     2.0, at 3× → 3.0, … bounded only by physics (you can't
            #     take fewer steps than the distance allows). Scaling by
            #     init_dist keeps it fair across spawns. The floor keeps a
            #     small reward for slow-but-successful crossings.
            #
            #     NOTE: because this is uncapped, a lucky very-fast crossing
            #     of a near gate can produce a large terminal bonus (e.g.
            #     4 m gate at 30 m/s → factor 3 → 3·success_bonus). PPO's
            #     advantage normalization absorbs moderate variance, but if
            #     value learning destabilizes, set `time_factor_max` in the
            #     config to re-introduce a soft ceiling (disabled when
            #     absent / null).
            #
            #   (Dropped vs. prior versions: align_factor — body-yaw
            #    alignment with target_yaw; and speed_factor — crossing
            #    velocity magnitude. Both pushed the policy toward "align
            #    then accelerate" strategies. We now want "reach the
            #    center fast, however you want to get there".)
            reward_cfg = self.cfg['reward']
            inner_half = float(self.cfg['gate']['size_interior']) / 2.0
            center_distance = math.sqrt(
                gate_lateral ** 2 + gate_vertical ** 2)
            # Floor the center factor so an off-center crossing still earns
            # `center_factor_min` of the centering credit instead of being
            # driven to 0 at the inner edge. Centering still matters (center
            # → 1.0) but no longer dominates the bonus, leaving room for the
            # steeper time_factor to make speed the primary driver.
            center_factor_min = float(reward_cfg.get('center_factor_min', 0.5))
            raw_center = max(0.0, 1.0 - center_distance / inner_half)
            center_factor = (
                center_factor_min + (1.0 - center_factor_min) * raw_center)
            ref_speed = float(reward_cfg.get('ref_speed', 10.0))
            time_target_min = int(reward_cfg.get('time_target_min', 20))
            time_factor_min = float(reward_cfg.get('time_factor_min', 0.1))
            control_hz = float(self.cfg['control_hz'])
            # Reference step count to fly init_dist straight at ref_speed,
            # floored so a near spawn can't make the unit reference tiny.
            ref_steps = max(
                float(time_target_min),
                self._initial_dist_to_target / ref_speed * control_hz)
            steps_taken = max(1, self._step_idx)
            # Speed ratio == avg_speed / ref_speed. A configurable exponent
            # steepens the curve symmetrically around 1.0: with exp>1, fast
            # crossings (ratio>1) are rewarded super-linearly AND slow ones
            # (ratio<1) are pushed down harder — both "reward fast" and
            # "penalize slow" at once. exp=1.0 reproduces the linear form.
            ratio = ref_steps / steps_taken
            time_factor_exp = float(reward_cfg.get('time_factor_exp', 1.0))
            time_factor = ratio ** time_factor_exp
            time_factor = max(time_factor_min, time_factor)
            # Optional soft ceiling — absent / null in config means uncapped.
            time_factor_max = reward_cfg.get('time_factor_max')
            if time_factor_max is not None:
                time_factor = min(float(time_factor_max), time_factor)

            bonus = (float(reward_cfg['success_bonus'])
                     * center_factor * time_factor)
            reward += bonus

            # Keep align/speed in info as diagnostics (no longer scaling
            # the bonus, but useful for inspecting what the policy WOULD
            # have scored under the old shape — handy when comparing
            # checkpoints across reward redesigns).
            align_factor = max(0.0, 1.0 - yaw_err_norm)
            speed_factor = omni_speed_score

            avg_speed = (self._initial_dist_to_target
                         / (steps_taken / control_hz))
            print(f'{self.drone_namespace} success bonus={bonus:.2f} '
                  f'(center={center_factor:.2f}, time={time_factor:.2f} '
                  f'@t={self._step_idx} steps, '
                  f'init_dist={self._initial_dist_to_target:.1f} m, '
                  f'avg_speed={avg_speed:.1f} m/s vs ref {ref_speed:.0f}  '
                  f'| diag align={align_factor:.2f} '
                  f'speed={speed_factor:.2f})')
            info['center_factor'] = center_factor
            info['time_factor'] = time_factor
            info['ref_steps'] = ref_steps
            info['avg_speed'] = avg_speed
            info['initial_dist_to_target'] = self._initial_dist_to_target
            info['align_factor'] = align_factor   # diagnostic, not in bonus
            info['speed_factor'] = speed_factor   # diagnostic, not in bonus
            info['terminal_bonus'] = bonus
        # Workspace OOB applies the full oob_penalty. A gate crash (either
        # frame hit or back-entry) applies the smaller crash_penalty — we
        # want a clear "don't do that" signal but milder than wandering off
        # into the void.
        if info.get('oob'):
            if info.get('crash'):
                reward -= float(self.cfg['reward'].get('crash_penalty', 0.0))
            else:
                reward -= float(self.cfg['reward']['oob_penalty'])

        self._step_idx += 1
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Curriculum hooks (called via VecEnv.env_method)
    # ------------------------------------------------------------------ #

    def set_v_max(self, v_max: float,
                  depth_tol: float | None = None) -> None:
        """Update v_max (and the gate depth_tol) at runtime.

        In ``speed`` mode v_max is the per-axis velocity action scale, so
        updating ``self.cfg['action']['v_max']`` takes effect immediately
        for action scaling, velocity obs normalization, the speed/vertical
        reward terms, and the crossing/forward speed scores.

        In ``rates`` mode the action does not use v_max for scaling, but
        v_max IS still used by the velocity observation normalization and
        the reward terms — so we still update it. depth_tol auto-scales in
        both modes (a high-v_max policy might still close on the gate fast
        enough to skip it).
        """
        v_max = float(v_max)
        self.cfg['action']['v_max'] = v_max
        if depth_tol is None:
            depth_tol = v_max * self._dt / 2.0 + 0.15
        self.cfg['gate']['depth_tol'] = float(depth_tol)
        tag = 'rates(obs-only)' if self._action_mode == 'rates' else 'speed'
        print(f'{self.drone_namespace} curriculum [{tag}]: '
              f'v_max={v_max:.2f} m/s, depth_tol={depth_tol:.3f} m')

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _sample_init_pose(self) -> tuple[np.ndarray, float]:
        """Sample drone init position as an OFFSET from the current target.

        ``init_bounds.x_min..x_max`` etc. are now interpreted as relative
        offsets from ``self._target_pos``, so the drone always spawns inside
        a fixed-shape box around its gate — independent of where the gate
        ended up after target_bounds randomization. Drone yaw is sampled
        uniformly in [-π, π] (absolute).
        """
        ib = self.cfg['init_bounds']
        rng = self.np_random
        rel_pos = np.array([
            rng.uniform(ib['x_min'], ib['x_max']),
            rng.uniform(ib['y_min'], ib['y_max']),
            rng.uniform(ib['z_min'], ib['z_max']),
        ])
        pos = self._target_pos + rel_pos
        yaw = float(rng.uniform(-math.pi, math.pi))
        return pos, yaw

    def _sample_target_pose(self) -> tuple[np.ndarray, float]:
        """Sample a fresh (position, yaw) target.

        If ``target_bounds`` is present in the config, the target is sampled
        uniformly from that box (with yaw in [yaw_min, yaw_max]). Otherwise
        falls back to the fixed ``target`` block — back-compat for fixed-goal
        evaluation. With target sampling enabled, the relative-yaw obs channel
        (drone_yaw − target_yaw)/π becomes a load-bearing signal instead of an
        alias for the world heading.
        """
        tb = self.cfg.get('target_bounds')
        if tb is not None:
            rng = self.np_random
            pos = np.array([
                rng.uniform(tb['x_min'], tb['x_max']),
                rng.uniform(tb['y_min'], tb['y_max']),
                rng.uniform(tb['z_min'], tb['z_max']),
            ])
            yaw_min = float(tb.get('yaw_min', -math.pi))
            yaw_max = float(tb.get('yaw_max', math.pi))
            yaw = float(rng.uniform(yaw_min, yaw_max))
            return pos, yaw
        # Fall back to fixed config target.
        tc = self.cfg['target']
        return (
            np.array([float(tc['x']), float(tc['y']), float(tc['z'])]),
            float(tc.get('yaw', 0.0)),
        )

    def _sample_exploring_start(self, init_pos: np.ndarray):
        """Sample a random initial attitude + velocity for exploring starts.

        Returns ``(roll, pitch, lin_vel_world, ang_vel_body)``. Forces the
        policy to experience aggressive high-speed / high-tilt regimes from
        step 0 so it discovers that high pitch/roll → high speed (instead of
        overfitting to the gentle-hover basin the conservative init biases
        it toward). Returns all-zeros (≡ original behavior) when the
        ``exploring_starts`` config block is absent or all maxima are 0.

        - lin_vel: magnitude uniform in [0, lin_vel_max], random isotropic
          direction (world frame). Random direction (not biased toward the
          gate) is intentional — the point is broad coverage of fast states
          and learning to redirect, not a head start. Each axis component is
          then capped so the drone keeps at least ``min_reaction_time``
          seconds before crossing the nearest OOB face on that axis (given
          ``init_pos``) — this removes the unwinnable spawns where a drone
          launched near the boundary is OOB before it can possibly recover,
          while leaving inward/tangential velocity untouched.
        - roll/pitch: each uniform in [-tilt_max, tilt_max].
        - ang_vel: each axis uniform in [-ang_vel_max, ang_vel_max] (body).
          Not capped — angular velocity doesn't drive translational OOB.
        """
        es = self.cfg.get('exploring_starts')
        if not es:
            return 0.0, 0.0, None, None
        rng = self.np_random
        lin_vel_max = float(es.get('lin_vel_max', 0.0))
        tilt_max = float(es.get('tilt_max', 0.0))
        ang_vel_max = float(es.get('ang_vel_max', 0.0))
        min_reaction_time = float(es.get('min_reaction_time', 0.5))

        lin_vel = None
        if lin_vel_max > 0.0:
            speed = float(rng.uniform(0.0, lin_vel_max))
            d = rng.normal(size=3)
            n = float(np.linalg.norm(d))
            d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            lin_vel = speed * d
            # Cap each component by the OOB margin in its direction so the
            # drone always has >= min_reaction_time to correct before OOB.
            if min_reaction_time > 0.0:
                offset = np.asarray(init_pos, dtype=np.float64) - self._target_pos
                ext = (self._oob_x_ext, self._oob_y_ext, self._oob_z_ext)
                for i in range(3):
                    margin = (ext[i] - offset[i] if lin_vel[i] > 0.0
                              else ext[i] + offset[i])
                    v_cap = max(0.0, float(margin)) / min_reaction_time
                    lin_vel[i] = float(np.clip(lin_vel[i], -v_cap, v_cap))

        roll = float(rng.uniform(-tilt_max, tilt_max)) if tilt_max > 0 else 0.0
        pitch = float(rng.uniform(-tilt_max, tilt_max)) if tilt_max > 0 else 0.0

        ang_vel = None
        if ang_vel_max > 0.0:
            ang_vel = rng.uniform(-ang_vel_max, ang_vel_max, size=3)

        return roll, pitch, lin_vel, ang_vel

    def _teleport(self, position: np.ndarray, yaw: float,
                  roll: float = 0.0, pitch: float = 0.0,
                  lin_vel: np.ndarray | None = None,
                  ang_vel: np.ndarray | None = None) -> None:
        """Hot-teleport the simulator to a kinematic state.

        roll/pitch default to 0 (yaw-only attitude, original behavior).
        lin_vel (world frame, m/s) and ang_vel (body frame, rad/s) default
        to zero. The exploring-starts path in reset() passes non-zero
        values so the policy spawns already moving / tilted.
        """
        req = SetPlatformState.Request()
        req.pose = Pose()
        req.pose.position.x = float(position[0])
        req.pose.position.y = float(position[1])
        req.pose.position.z = float(position[2])
        if roll == 0.0 and pitch == 0.0:
            qw, qx, qy, qz = _yaw_to_quat(yaw)
        else:
            qw, qx, qy, qz = _euler_to_quat(roll, pitch, yaw)
        req.pose.orientation.w = qw
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        lv = Vector3()
        if lin_vel is not None:
            lv.x = float(lin_vel[0])
            lv.y = float(lin_vel[1])
            lv.z = float(lin_vel[2])
        req.linear_velocity = lv
        av = Vector3()
        if ang_vel is not None:
            av.x = float(ang_vel[0])
            av.y = float(ang_vel[1])
            av.z = float(ang_vel[2])
        req.angular_velocity = av
        req.reset_to_hover = False

        future = self._set_state_client.call_async(req)
        # The svc_executor (separate thread) will resolve the future. Block
        # here until done or timeout.
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.005)
        if not future.done():
            raise RuntimeError('Teleport service call timed out.')
        result = future.result()
        if result is None or not result.success:
            msg = result.message if result is not None else 'no response'
            raise RuntimeError(f'Teleport service rejected the request: {msg}')

    def _send_speed_command(self, vx: float, vy: float,
                            vz: float, vyaw: float) -> None:
        # Body-frame velocity command (REP-103 base_link: x=forward, y=left,
        # z=up). Pairs with the body-frame observation.
        self.drone.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [vx, vy, vz], f'{self.drone_namespace}/base_link', vyaw)

    def _send_rates_command(self, roll_rate: float, pitch_rate: float,
                            yaw_rate: float, thrust: float) -> None:
        """Publish thrust (N) + body-rate (rad/s) directly to the platform's
        actuator-command topics.

        Bypasses the motion controller — the PID controller doesn't accept
        ACRO as input, and we already switched the platform to ACRO +
        YAW_SPEED in `_takeoff_and_engage`, so the platform consumes these
        topics as raw acro setpoints (see `processCommand` in
        as2_platform_multirotor_simulator.cpp's ACRO case).
        """
        stamp = self._svc_node.get_clock().now().to_msg()

        thrust_msg = Thrust()
        thrust_msg.header.stamp = stamp
        thrust_msg.header.frame_id = f'{self.drone_namespace}/base_link'
        thrust_msg.thrust = float(thrust)
        self._thrust_pub.publish(thrust_msg)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = stamp
        twist_msg.header.frame_id = f'{self.drone_namespace}/base_link'
        twist_msg.twist.angular.x = float(roll_rate)
        twist_msg.twist.angular.y = float(pitch_rate)
        twist_msg.twist.angular.z = float(yaw_rate)
        self._twist_pub.publish(twist_msg)

    def _send_motor_command(self, motor_speeds) -> None:
        """Publish the 4 motor angular velocities (rad/s) to the platform's
        actuator_command/motors topic. The patched platform routes these
        straight into the simulator's MOTOR_W mode — no inner controller."""
        msg = Actuators()
        msg.header.stamp = self._svc_node.get_clock().now().to_msg()
        msg.header.frame_id = f'{self.drone_namespace}/base_link'
        msg.velocity = [float(w) for w in motor_speeds]
        self._motors_pub.publish(msg)

    def _publish_gate_marker(self) -> None:
        """Publish a MESH_RESOURCE Marker for the cvar_gate at the configured
        target pose. Periodic so RViz picks it up after reconnects."""
        if self._gate_marker_pub is None:
            return
        m = Marker()
        m.header.frame_id = 'earth'
        m.header.stamp = self._svc_node.get_clock().now().to_msg()
        m.ns = 'rl_target_gate'
        m.id = self._gate_marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = (
            'package://as2_gazebo_assets/models/cvar_gate/meshes/model.dae')
        m.mesh_use_embedded_materials = True
        m.pose.position.x = float(self._target_pos[0])
        m.pose.position.y = float(self._target_pos[1])
        # cvar_gate mesh origin is at the BOTTOM of the gate (≈2.7 m tall),
        # so offset the marker down by half its height to center the opening
        # on the target.
        m.pose.position.z = float(self._target_pos[2]) - 2.7 / 2.0
        # Flip mesh by π so the decorated face aligns with the +target_yaw
        # crossing direction the reward expects (visualization only).
        qw, qx, qy, qz = _yaw_to_quat(self._target_yaw + math.pi)
        m.pose.orientation.w = qw
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        self._gate_marker_pub.publish(m)

    def _set_physics(self, pause: bool) -> None:
        """Pause or resume this drone's physics integration.

        Both directions wait for the service to confirm completion. The
        clock-publisher side is idempotent — calling pause-while-paused or
        resume-while-running returns success immediately — so no caching
        is needed on this side. Previous versions cached the local state
        and used fire-and-forget for resume; that allowed a single dropped
        or reordered resume to leave the simulator silently stuck paused,
        which manifests mid-episode as the agent freezing while the
        IMU/state-pub sensors keep publishing the last integrated state.
        """
        req = SetBool.Request()
        req.data = bool(pause)

        # One retry: if the first request times out without a response, the
        # platform might or might not have applied it. Sending a fresh
        # request is safe (server is idempotent) and converges us to the
        # requested state.
        for attempt in (0, 1):
            future = self._pause_client.call_async(req)
            deadline = time.time() + 1.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.005)
            if future.done() and future.result() is not None \
                    and future.result().success:
                self._physics_paused = bool(pause)
                return
            # Cancel the unresolved future before retrying so we don't leak
            # callbacks in the rclpy client's internal map.
            try:
                self._pause_client.remove_pending_request(future)
            except Exception:
                pass
            if attempt == 0:
                print(
                    f'[{self.drone_namespace}] _set_physics('
                    f'{"pause" if pause else "resume"}) timed out; retrying',
                    flush=True)

        # Both attempts failed — surface it rather than silently desyncing.
        raise RuntimeError(
            f'/sim_clock_publisher/pause_physics did not confirm '
            f'{"pause" if pause else "resume"} after 2 attempts. The '
            f'simulator is likely stuck — restart sim_clock_publisher_node '
            f'on this drone\'s ROS_DOMAIN_ID.')

    def _read_pose(self) -> tuple[np.ndarray, float]:
        pos = np.asarray(self.drone.position, dtype=np.float64)
        yaw = float(self.drone.orientation[2])
        return pos, yaw

    def _pose_quat_callback(self, msg: PoseStamped) -> None:
        """Cache the attitude quaternion from self_localization/pose (the same
        topic DroneInterface uses, but keeping the quaternion)."""
        q = msg.pose.orientation
        self._latest_quat = (q.w, q.x, q.y, q.z)

    def _twist_omega_callback(self, msg: TwistStamped) -> None:
        """Cache the body-frame angular velocity (measured body rates ω) from
        self_localization/twist (DroneInterface keeps only the linear part)."""
        w = msg.twist.angular
        self._latest_omega = (w.x, w.y, w.z)

    def _read_omega(self) -> np.ndarray:
        """Measured body rates ω (rad/s, body frame). Zeros until the first
        twist message arrives."""
        w = self._latest_omega
        if w is None:
            return np.zeros(3, dtype=np.float64)
        return np.array(w, dtype=np.float64)

    def _motor_speed_callback(self, msg: Actuators) -> None:
        """Cache the actual motor angular velocities from the platform's
        motor_speed topic (motor mode obs). Zeros-fallback until first message."""
        v = msg.velocity
        if len(v) >= 4:
            self._latest_motor_speed = (v[0], v[1], v[2], v[3])

    def _read_motor_speed(self) -> np.ndarray:
        """Actual motor angular velocities (rad/s). Zeros until the first
        motor_speed message arrives."""
        w = self._latest_motor_speed
        if w is None:
            return np.zeros(4, dtype=np.float64)
        return np.array(w, dtype=np.float64)

    def _read_rotation(self) -> np.ndarray:
        """Body→world rotation R_wb. Prefers the cached attitude quaternion
        (exact, singularity-free); falls back to reconstructing it from the
        DroneInterface euler if no pose has arrived yet (startup gap), which
        matches the previous euler-based behaviour during that window."""
        q = self._latest_quat
        if q is not None:
            return _quat_to_rotmat(q[0], q[1], q[2], q[3])
        roll, pitch, yaw = (float(a) for a in self.drone.orientation)
        return _quat_to_rotmat(*_euler_to_quat(roll, pitch, yaw))

    def _read_velocity(self) -> np.ndarray:
        """World-frame linear velocity.

        ``DroneInterface.speed`` returns the linear twist from
        ``self_localization/twist``, which is published in the **body** frame
        by the AS2 state estimator. The rest of the env (gate_normal_dot_vel,
        the world → body rotation that produces vel_body_x for the speed
        bonus, the workspace OOB check, etc.) assumes world-frame velocity,
        so we apply the yaw-only body → world rotation here.
        """
        vel_body = np.asarray(self.drone.speed, dtype=np.float64)
        yaw = float(self.drone.orientation[2])
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        return np.array([
            cos_y * vel_body[0] - sin_y * vel_body[1],
            sin_y * vel_body[0] + cos_y * vel_body[1],
            float(vel_body[2]),
        ])

    def _compute_state(self) -> SimpleNamespace:
        """Read pose+velocity once and compute every intermediate value used
        by both the observation and the reward. Returned as a SimpleNamespace
        so callers can do attribute access (``s.obs``, ``s.rel_body_x``…)
        without each one re-rotating the world into the body frame.

        Physics is paused at every call site (`_observe_step` is invoked
        after the pause step in `step()`; `_get_obs` is called from `reset()`
        which also pauses), so the underlying state doesn't shift between
        the read and use.
        """
        pos, yaw = self._read_pose()
        vel_world = self._read_velocity()
        rel_world = self._target_pos - pos
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        # World → body (yaw-only) rotation. REP-103: body x=forward, y=left.
        rel_body_x = cos_y * rel_world[0] + sin_y * rel_world[1]
        rel_body_y = -sin_y * rel_world[0] + cos_y * rel_world[1]
        vel_body_x = cos_y * vel_world[0] + sin_y * vel_world[1]
        vel_body_y = -sin_y * vel_world[0] + cos_y * vel_world[1]
        vel_body_z = float(vel_world[2])
        v_max = float(self.cfg['action']['v_max'])
        yaw_rel = _wrap_to_pi(yaw - self._target_yaw)
        # Bearing-to-target: atan2 of the target's body-frame position.
        # bearing = 0 → target directly in front; +π/2 → to the left; ±π →
        # behind; -π/2 → to the right. This replaced yaw_rel (drone yaw
        # vs gate yaw) in obs[6] when the reward stopped paying for body
        # yaw alignment — the policy now needs "where is the target
        # relative to my nose" instead of "am I facing the gate's
        # forward". The bearing channel stays useful even when the
        # position obs[0..1] saturate to ±1 at far distances; the angle
        # is exact regardless of how far the target is.
        bearing = math.atan2(rel_body_y, rel_body_x)  # yaw-only azimuth (diag)
        # Roll / pitch only consumed by the rates-mode observation. Reading
        # them unconditionally keeps the state namespace uniform between
        # modes (debug code can rely on s.roll / s.pitch always existing).
        roll = float(self.drone.orientation[0])
        pitch = float(self.drone.orientation[1])
        # Body→world rotation from the cached attitude QUATERNION (singularity-
        # free; see _read_rotation). Single attitude source for the camera
        # pointing and the 6D rotation below.
        r_wb = self._read_rotation()
        # Camera line-of-sight as a UNIT vector in the camera basis
        # [fwd, right, up] — continuous everywhere (no atan2 wrap when the gate
        # passes 90°/behind), goes straight into the obs. Plus the roll-invariant
        # off-axis angle for the reward. See `_camera_pointing`.
        los_cam, cam_off_axis = _camera_pointing(
            rel_world, r_wb, self._camera_pitch)
        cam_azimuth = math.atan2(los_cam[1], los_cam[0])   # diag only (from uvec)
        cam_elevation = math.atan2(los_cam[2], los_cam[0])  # diag only (from uvec)
        # Pointing error used by the REWARD is computed against the HORIZONTAL
        # line-of-sight (vertical component of rel_world zeroed). This makes the
        # "look at the gate" reward a function of ATTITUDE only: to shrink it the
        # drone must PITCH (plus yaw/roll) to swing the up-tilted optical axis
        # onto the gate's horizontal bearing. Changing z-position does NOT move
        # the horizontal LOS, so the policy can no longer satisfy the camera by
        # flying BELOW the gate (the descend exploit) — it must tilt. (Roll
        # credit is preserved: still the full-3-D off-axis, just vs a horizontal
        # target.)
        rel_world_h = np.array([rel_world[0], rel_world[1], 0.0])
        _, cam_off_axis_reward = _camera_pointing(
            rel_world_h, r_wb, self._camera_pitch)
        obs_list = [
            np.clip(rel_body_x / self._pos_max_x, -1.0, 1.0),
            np.clip(rel_body_y / self._pos_max_y, -1.0, 1.0),
            np.clip(rel_world[2] / self._pos_max_z, -1.0, 1.0),
            # tanh instead of clip on velocity: the drone can comfortably
            # exceed v_max in rates mode (no PID cap), and a hard clip kills
            # the gradient for the very fast states where we most want the
            # policy to react. tanh keeps obs in (-1, 1), gives ≈ 0.76 at
            # v=v_max, and still ≈ 0.995 at v=3·v_max — every speed has a
            # unique, differentiable value. Same v_max is used for shape;
            # bumping v_max stretches the "fast vs cruising" boundary.
            np.tanh(vel_body_x / v_max),
            np.tanh(vel_body_y / v_max),
            np.tanh(vel_body_z / v_max),
            # Gate line-of-sight unit vector in the camera frame [fwd, right,
            # up]: continuous on the sphere (no wrap), tells the policy where the
            # gate is in frame; the reward penalizes the off-axis angle acos(fwd).
            float(los_cam[0]),
            float(los_cam[1]),
            float(los_cam[2]),
        ]
        if self._action_mode in ('rates', 'motor'):
            # Low-level modes share this block. 6D continuous rotation: the
            # drone's attitude relative to the GATE FRAME (first two columns of
            # the gate→body rotation). Stable per-episode reference, so it
            # changes only when the drone rotates — never with its position.
            # Singularity-free (quaternion-sourced): no yaw wrap, no inversion
            # ambiguity, no pitch gimbal. Being gate-relative, it ALSO carries
            # the gate facing / which-side. See `_rotation_6d`.
            obs_list.extend(_rotation_6d(r_wb, self._target_yaw).tolist())
            # Measured body rates ω (rad/s, body frame), tanh-scaled. Current
            # rotational state — the policy must see it to regulate its rates
            # (and it is what the body-rate reward penalty acts on). Distinct
            # from the COMMAND (the previous action below).
            omega = self._read_omega()
            obs_list.extend([
                math.tanh(omega[0] / self._w_obs_scale),
                math.tanh(omega[1] / self._w_obs_scale),
                math.tanh(omega[2] / self._w_obs_scale),
            ])
            # Previous action u_{t-1} (rate/thrust or motor command), in [-1, 1].
            # Paired with the action-smoothness reward ‖u_t − u_{t-1}‖.
            obs_list.extend(float(a) for a in self._last_action)
        if self._action_mode == 'motor':
            # Actual (lagged) motor angular velocities, normalized to [-1, 1].
            # Makes the actuator's hidden spool-up state OBSERVABLE → the
            # motor-lag (tau) POMDP becomes Markov (the winner's key fix).
            mw = self._read_motor_speed()
            obs_list.extend(
                float(2.0 * w / self._motor_obs_scale - 1.0) for w in mw)
        elif self._action_mode == 'speed':
            # Speed mode doesn't control attitude; only relative heading matters.
            obs_list.append(yaw_rel / math.pi)
        obs = np.array(obs_list, dtype=np.float32)
        return SimpleNamespace(
            obs=obs,
            pos=pos,
            yaw=yaw,
            roll=roll,
            pitch=pitch,
            vel_world=vel_world,
            rel_world=rel_world,
            cos_y=cos_y,
            sin_y=sin_y,
            rel_body_x=rel_body_x,
            rel_body_y=rel_body_y,
            vel_body_x=vel_body_x,
            vel_body_y=vel_body_y,
            vel_body_z=vel_body_z,
            v_max=v_max,
            yaw_rel=yaw_rel,
            bearing=bearing,
            cam_azimuth=cam_azimuth,
            cam_elevation=cam_elevation,
            cam_off_axis=cam_off_axis,
            cam_off_axis_reward=cam_off_axis_reward,
            los_cam=los_cam,
            omega=self._read_omega(),
            last_action=np.asarray(self._last_action, dtype=np.float64),
            action_prev=np.asarray(self._action_prev, dtype=np.float64),
            motor_speed=self._read_motor_speed(),
        )

    def _get_obs(self) -> np.ndarray:
        """Body-frame observation. Dimensionality depends on action.mode.
        Shared core [0..8]: [0..2] (body_x, body_y, body_z) to target;
          [3..5] (vx, vy, vz) tanh-normalized by v_max; [6..8] camera LOS unit
          vector [fwd, right, up] in the camera frame (gate direction in view;
          continuous on the sphere — no atan2 wrap when the gate goes behind).
        Then:
          speed (10-D): + [9] relative yaw (drone − target) / π.
          rates (22-D): + [9..14] 6D continuous rotation — first two columns of
            the gate→body rotation (attitude relative to the stable gate frame,
            carries gate facing / which-side, quaternion-sourced, discontinuity-
            free); + [15..17] measured body rates ω (gyro, body frame, tanh-
            scaled); + [18..21] previous action (command u_{t-1}, in [-1, 1]).
          motor (26-D): the rates-mode 22-D + [22..25] actual motor speeds
            (rad/s, normalized to [-1, 1]) — direct per-motor control with the
            actuator state observable. See `_rotation_6d` / `_camera_pointing`.
        """
        return self._compute_state().obs

    def _compute_reward(self, dist_xy_norm: float, z_err_norm: float,
                        bearing_norm: float, yaw_err_norm: float,
                        forward_speed_score: float,
                        vertical_speed_norm: float) -> float:
        """Per-step reward (continuous):
        r = -k_pos·dist_xy_norm
            - k_height·z_err_norm
            - k_bearing·bearing_norm
            - k_yaw_align·yaw_err_norm
            + k_speed·forward_speed_score
            - k_vertical·vertical_speed_norm
        All inputs already normalized to [0, 1].
        """
        rw = self.cfg['reward']
        return (
            -float(rw['k_pos']) * dist_xy_norm
            - float(rw['k_height']) * z_err_norm
            - float(rw['k_bearing']) * bearing_norm
            - float(rw['k_yaw_align']) * yaw_err_norm
            + float(rw['k_speed']) * forward_speed_score
            - float(rw['k_vertical']) * vertical_speed_norm
        )

    def _check_done(self, drone_pos: np.ndarray, gate_depth: float,
                    gate_lateral: float, gate_vertical: float,
                    gate_normal_dot_vel: float
                    ) -> tuple[bool, bool, dict]:
        """Terminate on:
        - success: drone in the gate plane AND inside the inner opening AND
                   moving in the +target_yaw direction (legit forward cross)
        - crash (oob): one of
              (a) drone in the gate plane AND inside the annular frame region
                  (hit the gate frame between inner and outer square)
              (b) drone in the gate plane AND inside the inner opening BUT
                  moving in the −target_yaw direction (entering from behind)
        - oob: drone outside the workspace box (relative to current target)
        - timeout: max_steps reached
        """
        info: dict[str, Any] = {}
        gate = self.cfg['gate']
        inner_half = float(gate['size_interior']) / 2.0
        outer_half = float(gate['size_exterior']) / 2.0
        depth_tol = float(gate['depth_tol'])

        # Arm all gate-plane terminals once the drone is clearly AWAY from
        # the gate center (3D distance, not just off the plane). Any real
        # approach from a >= min_init_target_dist spawn must pass through
        # this shell, so it always arms; meanwhile a stale at-the-gate pose
        # after reset (dist ≈ 0) never arms. Using 3D distance rather than
        # |gate_depth| also handles the ~3%-of-episodes case where the drone
        # spawns inside the plane slab but laterally far — |depth| would be
        # small there and never arm, but the 3D distance is large.
        # arm_radius must satisfy: gate footprint < arm_radius <
        # min_init_target_dist, so it sits between "at the gate" and "at
        # spawn".
        arm_radius = float(gate.get('success_arm_radius', 2.0))
        dist_to_gate = float(np.linalg.norm(drone_pos - self._target_pos))
        if dist_to_gate >= arm_radius:
            self._success_armed = True

        if abs(gate_depth) < depth_tol:
            # In the plane slab but not yet armed → stale-pose phantom.
            # Suppress ALL gate terminals (success AND crash) and let the
            # episode continue so the real spawn pose can propagate.
            if not self._success_armed:
                info['phantom_suppressed'] = True
                return False, False, info
            in_plane_max = max(abs(gate_lateral), abs(gate_vertical))
            if in_plane_max < inner_half:
                if gate_normal_dot_vel > 0.0:
                    info['success'] = True
                    print(f'{self.drone_namespace} SUCCESS at step {self._step_idx}: '
                          f'depth={gate_depth:+.3f} lat={gate_lateral:+.3f} '
                          f'vert={gate_vertical:+.3f}')
                    return True, False, info
                # Through-the-opening but moving against the gate normal:
                # this is "entering from behind" — treat as a crash.
                info['oob'] = True
                info['crash'] = True
                info['back_entry'] = True
                print(f'{self.drone_namespace} CRASHED back-entry at step '
                      f'{self._step_idx}: normal·vel={gate_normal_dot_vel:+.3f} '
                      f'lat={gate_lateral:+.3f} vert={gate_vertical:+.3f}')
                return True, False, info
            if in_plane_max < outer_half:
                # Hit the gate frame's annular region — crash.
                info['oob'] = True
                info['crash'] = True
                print(f'{self.drone_namespace} CRASHED gate frame at step '
                      f'{self._step_idx}: depth={gate_depth:+.3f} '
                      f'lat={gate_lateral:+.3f} vert={gate_vertical:+.3f}')
                return True, False, info

        # Symmetric target-relative OOB: drone must stay within ±_oob_*_ext
        # of the *current* target on every axis. The box translates with the
        # active target without inheriting any asymmetry from the original
        # config target position.
        rel = drone_pos - self._target_pos
        if (abs(rel[0]) > self._oob_x_ext
                or abs(rel[1]) > self._oob_y_ext
                or abs(rel[2]) > self._oob_z_ext):
            info['oob'] = True
            return True, False, info

        if self._step_idx + 1 >= self._max_steps:
            info['timeout'] = True
            return False, True, info

        return False, False, info
