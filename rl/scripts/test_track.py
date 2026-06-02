"""Evaluate a trained PPO policy on a multi-gate race track.

The drone is teleported by the env's ``reset()`` to a random start pose, then
the test script overrides the env's target to gate 0. On every step the
script feeds the (target-relative) observation to ``model.predict``, applies
the body-frame action, and watches for ``info['success']``. When success
triggers the *target switches to the next gate without resetting or
teleporting* — the drone keeps flying through. Episode ends when all gates
are passed, the drone goes OOB, or it stalls long enough to time out.

Usage:
    python -m rl.scripts.test_track --model rl/models/ppo_drone_final.zip
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.env.drone_goal_env import DroneGoalEnv, _yaw_to_quat


# Default 4-gate clockwise track at z = 1.75. Tune for your workspace.
DEFAULT_GATES: list[dict] = [
    {'x':  3.0, 'y':  0.0, 'z': 1.75, 'yaw':  0.0},
    {'x':  3.0, 'y':  3.0, 'z': 1.75, 'yaw':  math.pi / 2},
    {'x':  0.0, 'y':  3.0, 'z': 1.75, 'yaw':  math.pi},
    {'x':  0.0, 'y':  0.0, 'z': 1.75, 'yaw': -math.pi / 2},
]


def _publish_track_markers(env: DroneGoalEnv, gates: list[dict]) -> None:
    """Publish a Marker for every gate on /rl/target_gate.

    The env's own 1 Hz timer publishes just the active gate; we cancel it and
    take over so all gates are visible at once. Republished periodically by
    the test loop because RViz Marker QoS is Volatile.
    """
    from visualization_msgs.msg import Marker

    pub = env._gate_marker_pub
    if pub is None:
        pub = env._svc_node.create_publisher(Marker, '/rl/target_gate', 10)
        env._gate_marker_pub = pub
    for i, gate in enumerate(gates):
        m = Marker()
        m.header.frame_id = 'earth'
        m.header.stamp = env._svc_node.get_clock().now().to_msg()
        m.ns = 'rl_target_gate'
        m.id = i
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = (
            'package://as2_gazebo_assets/models/cvar_gate/meshes/model.dae')
        m.mesh_use_embedded_materials = True
        m.pose.position.x = float(gate['x'])
        m.pose.position.y = float(gate['y'])
        m.pose.position.z = float(gate['z']) - 2.7 / 2.0
        qw, qx, qy, qz = _yaw_to_quat(float(gate['yaw']))
        m.pose.orientation.w = qw
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        pub.publish(m)


def _set_target(env: DroneGoalEnv, gate: dict) -> None:
    """Point the env at a new gate without teleporting the drone."""
    env._target_pos = np.array([gate['x'], gate['y'], gate['z']],
                               dtype=np.float64)
    env._target_yaw = float(gate['yaw'])
    env._step_idx = 0  # fresh per-gate budget so timeout doesn't fire mid-track


def _run_episode(env: DroneGoalEnv, model: PPO, gates: list[dict],
                 ep_idx: int, max_steps_per_gate: int,
                 deterministic: bool) -> dict:
    obs, _info = env.reset(seed=ep_idx)
    _set_target(env, gates[0])
    _publish_track_markers(env, gates)
    obs = env._get_obs()  # recompute obs against gate 0

    current_gate = 0
    episode_reward = 0.0
    total_steps = 0
    steps_in_gate = 0
    times_per_gate: list[int] = []
    bonuses: list[float] = []
    outcome = 'unknown'

    print(f'\n=== Episode {ep_idx + 1} ===')
    g0 = gates[0]
    print(f'Starting target: gate 0 -> ({g0["x"]:.2f}, {g0["y"]:.2f}, '
          f'{g0["z"]:.2f}), yaw={g0["yaw"]:.2f}')

    max_total = max_steps_per_gate * len(gates) + 200  # safety budget
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        env._apply_action(action)
        time.sleep(env._dt)
        obs, reward, terminated, truncated, info = env._observe_step()
        episode_reward += reward
        total_steps += 1
        steps_in_gate += 1

        if info.get('success'):
            bonus = float(info.get('terminal_bonus', 0.0))
            bonuses.append(bonus)
            times_per_gate.append(steps_in_gate)
            print(f'  [step {total_steps:4d}] PASSED gate {current_gate} '
                  f'in {steps_in_gate} steps  '
                  f'dist_xy={info["dist_xy"]:.3f}  '
                  f'bonus={bonus:.2f}  '
                  f"speed={info.get('speed_factor', 0.0):.2f}  "
                  f"align={info.get('align_factor', 0.0):.2f}  "
                  f"center={info.get('center_factor', 0.0):.2f}")
            current_gate += 1
            steps_in_gate = 0
            if current_gate >= len(gates):
                print(f'  Track complete in {total_steps} steps  '
                      f'episode_reward={episode_reward:.2f}')
                outcome = 'completed'
                break
            _set_target(env, gates[current_gate])
            obs = env._get_obs()
            gn = gates[current_gate]
            print(f'  -> switching to gate {current_gate} '
                  f'({gn["x"]:.2f}, {gn["y"]:.2f}, {gn["z"]:.2f}), '
                  f'yaw={gn["yaw"]:.2f}')
            # Republish so RViz keeps the markers visible.
            _publish_track_markers(env, gates)
            continue

        if info.get('oob'):
            print(f'  [step {total_steps:4d}] OUT OF BOUNDS while chasing '
                  f'gate {current_gate}  reward={episode_reward:.2f}')
            outcome = 'oob'
            break

        if truncated or steps_in_gate >= max_steps_per_gate:
            print(f'  [step {total_steps:4d}] TIMEOUT on gate {current_gate}  '
                  f'reward={episode_reward:.2f}')
            outcome = 'timeout'
            break

        if total_steps >= max_total:
            print(f'  [step {total_steps:4d}] HARD STOP (safety budget)  '
                  f'reward={episode_reward:.2f}')
            outcome = 'hard_stop'
            break

        # Republish markers every ~1 s so RViz keeps showing all gates.
        if total_steps % 10 == 0:
            _publish_track_markers(env, gates)

    return {
        'outcome': outcome,
        'gates_passed': current_gate,
        'total_steps': total_steps,
        'episode_reward': episode_reward,
        'times_per_gate': times_per_gate,
        'bonuses': bonuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the .zip PPO checkpoint to load.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to rl_config.yaml (defaults to rl/rl_config.yaml).')
    parser.add_argument('--namespace', type=str, default='drone0',
                        help='Drone namespace (must be running in the simulator).')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps-per-gate', type=int, default=300)
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions (default: deterministic).')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config) if args.config else here / 'rl_config.yaml'

    print(f'Loading PPO model from {args.model}')
    model = PPO.load(args.model)

    env = DroneGoalEnv(config_path=str(cfg_path), drone_namespace=args.namespace)
    # Cancel the env's single-gate marker timer; the test publishes all gates.
    if env._gate_marker_timer is not None:
        env._gate_marker_timer.cancel()

    try:
        summaries = []
        for ep in range(args.episodes):
            summary = _run_episode(
                env, model, DEFAULT_GATES, ep_idx=ep,
                max_steps_per_gate=args.max_steps_per_gate,
                deterministic=not args.stochastic)
            summaries.append(summary)

        print('\n=== Summary ===')
        completed = sum(1 for s in summaries if s['outcome'] == 'completed')
        print(f'completed: {completed}/{len(summaries)} episodes')
        for i, s in enumerate(summaries):
            print(f'  ep {i + 1}: outcome={s["outcome"]:<10s} '
                  f'gates={s["gates_passed"]}/{len(DEFAULT_GATES)}  '
                  f'steps={s["total_steps"]:4d}  '
                  f'reward={s["episode_reward"]:+.2f}  '
                  f'per_gate={s["times_per_gate"]}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
