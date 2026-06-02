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
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
import yaml
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_msgs.srv import SetPlatformState
from geometry_msgs.msg import Pose, Vector3
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))  # (w, x, y, z)


class DroneGoalEnv(gym.Env):
    """Drive the drone to a randomized (x, y, z, yaw) target."""

    metadata = {'render_modes': []}

    # Class-level flag so only the first env instance owns the gate Marker
    # publisher (avoids 20 drones each spamming the same marker topic).
    _gate_marker_owned = False

    def __init__(self, config_path: str | Path,
                 drone_namespace: str | None = None):
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
            use_sim_time=drone_cfg['use_sim_time'],
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
        # doesn't drift while the policy computes the next action.
        self._pause_client = self._svc_node.create_client(
            SetBool, f'/{self.drone_namespace}/pause_physics')
        if not self._pause_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                'pause_physics service is unavailable after 10s. Make sure '
                'the patched as2_platform_multirotor_simulator is running.'
            )
        self._physics_paused = False

        # Action: 4-D, body-frame velocity (vx, vy, vz) + yaw rate, normalized.
        # Observation: 7-D, body-frame relative position to target (3) +
        # body-frame drone velocity (3) + absolute drone yaw / π (1).
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

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
        self._last_action = np.zeros(4, dtype=np.float32)
        self._closed = False

        # Precomputed normalization divisors based on workspace edges relative
        # to the fixed config target. Used by _get_obs and _compute_reward so
        # every reward term lives in [0, 1] before its weight is applied.
        ws = self.cfg['workspace']
        tx = float(self.cfg['target']['x'])
        ty = float(self.cfg['target']['y'])
        tz = float(self.cfg['target']['z'])
        self._pos_max_x = max(abs(float(ws['x_max']) - tx), abs(float(ws['x_min']) - tx))
        self._pos_max_y = max(abs(float(ws['y_max']) - ty), abs(float(ws['y_min']) - ty))
        self._pos_max_z = max(abs(float(ws['z_max']) - tz), abs(float(ws['z_min']) - tz))
        self._max_dist_xy = math.sqrt(self._pos_max_x ** 2 + self._pos_max_y ** 2)

        self._takeoff_and_engage_speed_mode()

        # First env to construct owns the gate marker publisher. Republished
        # at 1 Hz so RViz picks it up regardless of when it connects (default
        # Marker display QoS is Volatile, so a single latched publish wouldn't
        # survive reconnects).
        self._gate_marker_pub = None
        self._gate_marker_timer = None
        if not DroneGoalEnv._gate_marker_owned:
            DroneGoalEnv._gate_marker_owned = True
            self._gate_marker_pub = self._svc_node.create_publisher(
                Marker, '/rl/target_gate', 10)
            self._gate_marker_timer = self._svc_node.create_timer(
                1.0, self._publish_gate_marker)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _takeoff_and_engage_speed_mode(self) -> None:
        if not self.drone.arm():
            raise RuntimeError('Failed to arm the drone.')
        if not self.drone.offboard():
            raise RuntimeError('Failed to switch to offboard.')
        if not self.drone.takeoff(
                height=float(self.cfg['takeoff_height']),
                speed=float(self.cfg['takeoff_speed'])):
            raise RuntimeError('Takeoff failed.')

        # Sending a zero-velocity SPEED command pins the platform in SPEED +
        # YAW_SPEED control mode for the rest of training.
        self._send_speed_command(0.0, 0.0, 0.0, 0.0)

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

        self._target_pos = self._sample_target_pose()

        min_dist = float(self.cfg.get('min_init_target_dist', 0.0))
        for _ in range(50):
            init_pos, init_yaw = self._sample_init_pose()
            if np.linalg.norm(init_pos - self._target_pos) >= min_dist:
                break

        self._teleport(init_pos, init_yaw)

        # Re-issue zero speed command so SPEED mode stays active.
        self._send_speed_command(0.0, 0.0, 0.0, 0.0)

        # Freeze physics immediately so the drone doesn't drift during the
        # state-publish settle sleep (or while waiting for the next action).
        # State publishers run on independent timers, so the DroneInterface
        # still receives the post-teleport pose before we read it.
        self._set_physics(True)
        time.sleep(2.0 * self._dt)

        self._step_idx = 0
        return self._get_obs(), {
            'target_pos': self._target_pos.copy(),
            'target_yaw': self._target_yaw,
            'init_pos': init_pos,
            'init_yaw': init_yaw,
        }

    def step(self, action: np.ndarray):
        # Single-env path: send command, sleep one control period, observe.
        # Vectorized swarm code reuses _apply_action / _observe_step directly
        # so the dt sleep happens once per VecEnv.step instead of per-env.
        self._apply_action(action)
        time.sleep(self._dt)
        return self._observe_step()

    def _apply_action(self, action: np.ndarray) -> None:
        """Send the velocity command for ``action``; do NOT sleep or observe."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = a
        v_max = float(self.cfg['action']['v_max'])
        yaw_rate_max = float(self.cfg['action']['yaw_rate_max'])
        vx, vy, vz = (a[:3] * v_max).tolist()
        vyaw = float(a[3] * yaw_rate_max)
        # Resume physics (no-op if not paused) before pushing the new command.
        self._set_physics(False)
        self._send_speed_command(vx, vy, vz, vyaw)

    def _observe_step(self):
        """Read pose+velocity, compute body-frame quantities, reward, and
        termination. Mirrors the ROS1 gate-crossing formulation:
        - position/velocity expressed in the drone's body frame (yaw-only)
        - bearing = atan2(body_y, body_x) penalizes "not facing the target"
        - forward-speed bonus encourages flying through, not creeping in
        - terminal bonus scales by bearing alignment AND forward speed
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
        obs = np.array([
            np.clip(rel_body_x / self._max_dist_xy, -1.0, 1.0),
            np.clip(rel_body_y / self._max_dist_xy, -1.0, 1.0),
            np.clip(rel_world[2] / self._pos_max_z, -1.0, 1.0),
            np.clip(vel_body_x / v_max, -1.0, 1.0),
            np.clip(vel_body_y / v_max, -1.0, 1.0),
            np.clip(vel_body_z / v_max, -1.0, 1.0),
            yaw / math.pi,
        ], dtype=np.float32)

        dist_xy = float(math.sqrt(rel_body_x ** 2 + rel_body_y ** 2))
        bearing = math.atan2(rel_body_y, rel_body_x)
        yaw_err = _wrap_to_pi(yaw - self._target_yaw)
        dist_xy_norm = dist_xy / self._max_dist_xy
        z_err_norm = abs(float(rel_world[2])) / self._pos_max_z
        bearing_norm = abs(bearing) / math.pi
        yaw_err_norm = abs(yaw_err) / math.pi
        # Forward-speed score: proportional and signed — only positive forward
        # body velocity is rewarded. Reversing through the gate (vx_body < 0)
        # gives 0 regardless of |v|, blocking the "back into the goal" exploit.
        # When yaw is aligned with target_yaw, +vx_body = +target_yaw world dir
        # = the legit gate-crossing direction.
        forward_speed_score = max(0.0, min(vel_body_x / v_max, 1.0))
        vertical_speed_norm = min(abs(vel_body_z) / v_max, 1.0)

        reward = self._compute_reward(
            dist_xy_norm, z_err_norm, bearing_norm, yaw_err_norm,
            forward_speed_score, vertical_speed_norm)
        terminated, truncated, info = self._check_done(pos, dist_xy)
        info.update({
            'dist_xy': dist_xy,
            'bearing': bearing,
            'yaw_err': yaw_err,
            'vel_body_x': vel_body_x,
            'vel_body_y': vel_body_y,
            'vel_body_z': vel_body_z,
            'dist_xy_norm': dist_xy_norm,
            'z_err_norm': z_err_norm,
            'bearing_norm': bearing_norm,
            'yaw_err_norm': yaw_err_norm,
            'forward_speed_score': forward_speed_score,
            'vertical_speed_norm': vertical_speed_norm,
        })
        if info.get('success'):
            # Gate-crossing-style terminal: full bonus only when aligned with
            # target orientation, flying through at v_max forward, and
            # crossing through the exact center of the gate.
            align_factor = max(0.0, 1.0 - yaw_err_norm)
            speed_factor = forward_speed_score  # positive forward only, [0, 1]
            pos_tol = float(self.cfg['tolerance']['pos_tol'])
            center_factor = max(0.0, 1.0 - dist_xy / pos_tol)
            bonus = (float(self.cfg['reward']['success_bonus'])
                     * align_factor * speed_factor * center_factor)
            reward += bonus
            print(f'{self.drone_namespace} success bonus={bonus:.2f} '
                  f'(align={align_factor:.2f}, speed={speed_factor:.2f}, '
                  f'center={center_factor:.2f})')
            info['align_factor'] = align_factor
            info['speed_factor'] = speed_factor
            info['center_factor'] = center_factor
            info['terminal_bonus'] = bonus
        if info.get('oob'):
            reward -= float(self.cfg['reward']['oob_penalty'])

        self._step_idx += 1
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _sample_init_pose(self) -> tuple[np.ndarray, float]:
        ib = self.cfg['init_bounds']
        rng = self.np_random
        pos = np.array([
            rng.uniform(ib['x_min'], ib['x_max']),
            rng.uniform(ib['y_min'], ib['y_max']),
            rng.uniform(ib['z_min'], ib['z_max']),
        ])
        yaw = float(rng.uniform(-math.pi, math.pi))
        return pos, yaw

    def _sample_target_pose(self) -> np.ndarray:
        tc = self.cfg['target']
        return np.array([float(tc['x']), float(tc['y']), float(tc['z'])])

    def _teleport(self, position: np.ndarray, yaw: float) -> None:
        req = SetPlatformState.Request()
        req.pose = Pose()
        req.pose.position.x = float(position[0])
        req.pose.position.y = float(position[1])
        req.pose.position.z = float(position[2])
        qw, qx, qy, qz = _yaw_to_quat(yaw)
        req.pose.orientation.w = qw
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.linear_velocity = Vector3()
        req.angular_velocity = Vector3()
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

    def _publish_gate_marker(self) -> None:
        """Publish a MESH_RESOURCE Marker for the cvar_gate at the configured
        target pose. Periodic so RViz picks it up after reconnects."""
        if self._gate_marker_pub is None:
            return
        m = Marker()
        m.header.frame_id = 'earth'
        m.header.stamp = self._svc_node.get_clock().now().to_msg()
        m.ns = 'rl_target_gate'
        m.id = 0
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
        qw, qx, qy, qz = _yaw_to_quat(self._target_yaw)
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

        Cached on ``_physics_paused`` so we skip the service call when the
        platform is already in the requested state. Pause calls block briefly
        so the freeze takes effect before subsequent code runs; resume calls
        are fire-and-forget to keep the per-step path latency-free.
        """
        if pause == self._physics_paused:
            return
        req = SetBool.Request()
        req.data = pause
        future = self._pause_client.call_async(req)
        if pause:
            deadline = time.time() + 1.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.005)
        self._physics_paused = pause

    def _read_pose(self) -> tuple[np.ndarray, float]:
        pos = np.asarray(self.drone.position, dtype=np.float64)
        yaw = float(self.drone.orientation[2])
        return pos, yaw

    def _read_velocity(self) -> np.ndarray:
        """World-frame linear velocity from DroneInterface (vx, vy, vz)."""
        return np.asarray(self.drone.speed, dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        """7-D body-frame observation, mirrors the ROS1 gate-crossing setup:
        (body_x, body_y, body_z) to target, (vx, vy, vz) in body frame,
        and the relative yaw (drone − target) normalized to [-1, 1].
        """
        pos, yaw = self._read_pose()
        vel = self._read_velocity()
        rel_world = self._target_pos - pos
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rel_body_x = cos_y * rel_world[0] + sin_y * rel_world[1]
        rel_body_y = -sin_y * rel_world[0] + cos_y * rel_world[1]
        vel_body_x = cos_y * vel[0] + sin_y * vel[1]
        vel_body_y = -sin_y * vel[0] + cos_y * vel[1]
        v_max = float(self.cfg['action']['v_max'])
        yaw_rel = _wrap_to_pi(yaw - self._target_yaw)
        obs = np.array([
            np.clip(rel_body_x / self._max_dist_xy, -1.0, 1.0),
            np.clip(rel_body_y / self._max_dist_xy, -1.0, 1.0),
            np.clip(rel_world[2] / self._pos_max_z, -1.0, 1.0),
            np.clip(vel_body_x / v_max, -1.0, 1.0),
            np.clip(vel_body_y / v_max, -1.0, 1.0),
            np.clip(vel[2] / v_max, -1.0, 1.0),
            yaw_rel / math.pi,
        ], dtype=np.float32)
        return obs

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

    def _check_done(self, drone_pos: np.ndarray,
                    dist_xy: float) -> tuple[bool, bool, dict]:
        info: dict[str, Any] = {}
        tol = self.cfg['tolerance']
        if dist_xy < float(tol['pos_tol']):
            info['success'] = True
            print(f'{self.drone_namespace} success at step {self._step_idx}: '
                  f'dist_xy={dist_xy:.3f}')
            return True, False, info

        ws = self.cfg['workspace']
        if (drone_pos[0] < ws['x_min'] or drone_pos[0] > ws['x_max']
                or drone_pos[1] < ws['y_min'] or drone_pos[1] > ws['y_max']
                or drone_pos[2] < ws['z_min'] or drone_pos[2] > ws['z_max']):
            info['oob'] = True
            return True, False, info

        if self._step_idx + 1 >= self._max_steps:
            info['timeout'] = True
            return False, True, info

        return False, False, info
