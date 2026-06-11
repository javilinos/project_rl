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


# Distance behind gate 0 (on the -target_yaw approach side) where the drone
# is teleported at the start of every episode. Same heading as gate 0's yaw,
# so the policy starts already pointed at the gate.
START_DIST_FROM_GATE0 = 2.0  # m


# Real circuit track ("MBZIRC-ish" layout). Gate centres come from the
# physical pose list:
#
#   gate01:               (12.5,  2.0, 1.45, π)
#   gate02:               ( 6.5,  6.0, 1.45, 2.35619)
#   gate03:               ( 5.5, 14.0, 1.45, 2.0944)
#   gate04:               ( 2.5, 24.0, 1.45, 1.5708)
#   gate05:               ( 7.5, 30.0, 1.45, -0.174533)
#   gate06:               (12.2, 22.0, 1.45, 0.0)
#   gate07_splitup:       (17.5, 30.0, 4.15, 1.39626)   ← elevated split branch
#   gate07_splitdown:     (17.5, 30.0, 1.45, -1.74533)  ← low split branch (default)
#   gate08:               (18.5, 22.0, 1.45, -1.39626)
#   gate09:               (20.5, 14.0, 1.45, -1.74533)
#   gate10_ladderup:      (18.5,  6.0, 4.15, -2.35619)  ← elevated ladder
#   gate10_ladderdown:    (18.5,  6.0, 1.45, -2.35619)  ← low ladder (default)
#
# Each pose is [x, y, z_opening_centre, yaw_rad]. z is the centre of the
# gate's inner opening; the marker publisher already offsets the mesh down
# by 2.7/2 = 1.35 m so the cvar_gate's bottom sits at z - 1.35 and the
# opening centre lands exactly at the target z (works for z = 1.45 and the
# elevated z = 4.15 gates alike).
#
# WORKSPACE NOTE: this track lives in x ∈ [2.5, 20.5], y ∈ [2, 30] —
# you'll need to bump `workspace.x_min/x_max/y_min/y_max` in rl_config.yaml
# (and probably target-relative `init_bounds`) so the drone doesn't
# immediately OOB at spawn.
#
# DEFAULT_GATES uses the LOW path through gate07 and the LOW ladder for
# gate10. The two ALT_GATES_WITH_HIGH_BRANCH dicts let you swap to the
# elevated branches; rebuild the list in main() if you want a different mix.
GATE_07_SPLITUP   = {'x': 17.5, 'y': 30.0, 'z': 4.00, 'yaw':  1.39626}
GATE_07_SPLITDOWN = {'x': 17.5, 'y': 30.0, 'z': 1.30, 'yaw': -1.74533}
GATE_10_LADDERUP   = {'x': 18.5, 'y':  6.0, 'z': 4.00, 'yaw': -2.35619}
GATE_10_LADDERDOWN = {'x': 18.5, 'y':  6.0, 'z': 1.30, 'yaw': -2.35619}
# 3 m alongside LADDERDOWN (offset along the gate's *lateral* axis, i.e.
# perpendicular to its forward direction, not world-x), same z, yaw rotated
# by π so the legitimate crossing direction flips. After passing ladderdown
# the drone carries momentum heading (-x, -y); to cross this gate it must
# decelerate, loop, and re-enter heading (+x, +y) — i.e., the loop-back
# maneuver that the stacked same-xy gate pair could not be cleared with.
#
# Lateral offset = ladderdown.yaw + π/2 = -135° + 90° = -45°
#   dx = 3 · cos(-45°) =  +2.121
#   dy = 3 · sin(-45°) =  -2.121
# → (18.5 + 2.121, 6.0 - 2.121) ≈ (20.62, 3.88)
GATE_10_LADDERDOWN_INV = {'x': 20.62, 'y':  3.88, 'z': 1.30, 'yaw':  0.78540,
                          'visible': False}

DEFAULT_GATES: list[dict] = [
    {'x': 12.5, 'y':  2.0, 'z': 1.45, 'yaw':  math.pi},         # gate01
    {'x':  6.5, 'y':  6.0, 'z': 1.45, 'yaw':  2.35619},         # gate02
    {'x':  5.5, 'y': 14.0, 'z': 1.45, 'yaw':  2.0944},          # gate03
    {'x':  2.5, 'y': 23.8, 'z': 1.45, 'yaw':  1.5708},          # gate04
    {'x':  7.5, 'y': 30.0, 'z': 1.45, 'yaw': -0.174533},        # gate05
    {'x': 12.2, 'y': 22.0, 'z': 1.45, 'yaw':  0.0},             # gate06
    GATE_07_SPLITUP,
    GATE_07_SPLITDOWN,                                          # gate07 (low branch)
    {'x': 18.5, 'y': 22.0, 'z': 1.45, 'yaw': -1.39626},         # gate08
    {'x': 20.5, 'y': 14.0, 'z': 1.45, 'yaw': -1.74533},         # gate09
    GATE_10_LADDERUP,
    GATE_10_LADDERDOWN_INV,                                     # gate10-inv (loop-back)
    GATE_10_LADDERDOWN,                                         # gate10 (low ladder)
]


# 8-gate counter-clockwise circular track at radius 5 m, z = 1.75 m. Each
# gate's yaw is tangent to the circle (θ + π/2), so flying straight forward
# through one gate aims you at the next. Mixes every yaw in [-π, π] — only
# meaningful once the policy has been retrained with randomized target_yaw,
# otherwise the policy will attack most gates from the wrong side as before.
# Switch on by changing the gates argument in main() to DEFAULT_GATES_2.
_TRACK_RADIUS = 5.0
_N_CIRCLE_GATES = 8
DEFAULT_GATES_2: list[dict] = [
    {
        'x': _TRACK_RADIUS * math.cos(2.0 * math.pi * i / _N_CIRCLE_GATES),
        'y': _TRACK_RADIUS * math.sin(2.0 * math.pi * i / _N_CIRCLE_GATES),
        'z': 1.75,
        'yaw': ((2.0 * math.pi * i / _N_CIRCLE_GATES + math.pi / 2.0
                 + math.pi) % (2.0 * math.pi)) - math.pi,
    }
    for i in range(_N_CIRCLE_GATES)
]


def _publish_track_markers(env: DroneGoalEnv, gates: list[dict]) -> None:
    """Publish all gates as a single MarkerArray on /rl/track_gates.

    A MarkerArray bundles every gate into one ROS message, so neither the
    publisher's queue nor RViz's subscriber depth can drop individual gates.
    The env's per-target single-Marker publisher (1 Hz, /rl/target_gate) is
    cancelled by the test driver before this is invoked, so the two streams
    don't fight over the visualization.
    """
    from visualization_msgs.msg import Marker, MarkerArray

    pub = getattr(env, '_track_marker_pub', None)
    if pub is None:
        pub = env._svc_node.create_publisher(MarkerArray, '/rl/track_gates', 10)
        env._track_marker_pub = pub

    array = MarkerArray()
    stamp = env._svc_node.get_clock().now().to_msg()
    for i, gate in enumerate(gates):
        # Per-gate opt-out: gates that only exist to shape the policy's
        # trajectory (e.g. the LADDERDOWN_INV loop-back waypoint) set
        # `visible: False` so the test still treats them as targets but
        # they don't clutter the RViz track view.
        if not gate.get('visible', True):
            continue
        m = Marker()
        m.header.frame_id = 'earth'
        m.header.stamp = stamp
        m.ns = 'rl_track_gates'
        m.id = i
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = (
            'package://as2_gazebo_assets/models/cvar_gate/meshes/model.dae')
        m.mesh_use_embedded_materials = True
        m.pose.position.x = float(gate['x'])
        m.pose.position.y = float(gate['y'])
        m.pose.position.z = float(gate['z']) - 2.7 / 2.0
        # The cvar_gate mesh's decorated front face is on -x of its local
        # frame, but the reward defines "+target_yaw direction" as the legit
        # crossing direction. Rotate the marker by π so the visual front
        # matches the drone's correct crossing side. Pure visualization fix
        # — does NOT change the gate yaw used by the policy / reward.
        qw, qx, qy, qz = _yaw_to_quat(float(gate['yaw']) + math.pi)
        m.pose.orientation.w = qw
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        array.markers.append(m)
    pub.publish(array)


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

    # Override the env's random init: teleport the drone START_DIST_FROM_GATE0
    # meters in front of gate 0 (i.e., on the -target_yaw side), already
    # facing the crossing direction. Physics was paused by env.reset(); we
    # don't unpause here — the first env.step's _apply_action handles that.
    g0 = gates[0]
    cos_t = math.cos(g0['yaw'])
    sin_t = math.sin(g0['yaw'])
    start_pos = np.array([
        g0['x'] - START_DIST_FROM_GATE0 * cos_t,
        g0['y'] - START_DIST_FROM_GATE0 * sin_t,
        g0['z'],
    ])
    env._teleport(start_pos, float(g0['yaw']))
    # Re-issue a zero-effort hold so the platform's "last reference" tracks
    # the just-teleported pose. Must match env.action_mode:
    #   speed → 0 velocity + 0 yaw_rate (motion-controller stays in SPEED).
    #   rates → hover thrust + 0 angular rates. Sending the speed flavor
    #           here would route through motion_ref_handler and try to
    #           renegotiate SPEED on the platform that's in ACRO from init,
    #           which freezes the simulator within a few hundred ms — same
    #           failure mode reset() had before we mode-dispatched there.
    if env._action_mode == 'speed':
        env._send_speed_command(0.0, 0.0, 0.0, 0.0)
    else:
        env._send_rates_command(0.0, 0.0, 0.0, env._thrust_hover)
    # Let the state publishers catch up so _get_obs reads the new pose.
    time.sleep(0.2)

    _publish_track_markers(env, gates)
    obs = env._get_obs()  # observation against gate 0 from the fixed start

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
        # env.step() handles control-period timing with overhead compensation
        # so /drone0/motion_reference/twist publishes at control_hz.
        obs, reward, terminated, truncated, info = env.step(action)
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
    parser.add_argument('--max-steps-per-gate', type=int, default=1024)
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
