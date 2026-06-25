"""Fly a TU-Delft MonoRace PPO policy on the test_track, two backends:

  --backend sim  (default): pure-C++ multirotor_pysim (ROS-free, fixed dt=0.01,
                  deterministic). No simulator process needed.
  --backend ros : the Aerostack2 platform (needs the sim running with the patched
                  multirotor platform + uav_config_cvar_racing.yaml + motor mode).

    python -m rl.scripts.run_quadrace_policy --backend sim \
        --model .../models/TT/256000000.zip --motor-perm 3,0,1,2

The policy starts on the FLOOR behind gate 0 and flies up itself; on each gate
pass the target advances to the next gate.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.env.track import GATES_ENU


def publish_track_markers(env, gates):
    """Publish all gates as one MarkerArray on /rl/track_gates (earth frame).
    ROS backend only. The cvar_gate mesh origin is at its bottom, so the marker
    is dropped 2.7/2 m to centre the opening on the gate z; the mesh is yawed +pi
    so its decorated face points along the crossing direction."""
    from visualization_msgs.msg import Marker, MarkerArray
    from rl.env.drone_goal_env import _yaw_to_quat
    pub = getattr(env, '_track_pub', None)
    if pub is None:
        pub = env._svc_node.create_publisher(MarkerArray, '/rl/track_gates', 10)
        env._track_pub = pub
    arr = MarkerArray()
    stamp = env._svc_node.get_clock().now().to_msg()
    for i, g in enumerate(gates):
        m = Marker()
        m.header.frame_id = 'earth'
        m.header.stamp = stamp
        m.ns = 'quadrace_gates'
        m.id = i
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = (
            'package://as2_gazebo_assets/models/cvar_gate/meshes/model.dae')
        m.mesh_use_embedded_materials = True
        m.pose.position.x = float(g['x'])
        m.pose.position.y = float(g['y'])
        m.pose.position.z = float(g['z']) - 2.7 / 2.0
        qw, qx, qy, qz = _yaw_to_quat(float(g['yaw']) + math.pi)
        m.pose.orientation.w = qw
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.scale.x = m.scale.y = m.scale.z = 1.0
        arr.markers.append(m)
    pub.publish(arr)


def _passed(prev_xyz, cur_xyz, gate, tol):
    """True if the segment prev->cur crossed the gate plane within tolerance."""
    n = np.array([math.cos(gate['yaw']), math.sin(gate['yaw'])])
    g = np.array([gate['x'], gate['y']])
    prev_p = float(np.dot(prev_xyz[:2] - g, n))
    cur_p = float(np.dot(cur_xyz[:2] - g, n))
    if not (prev_p < 0.0 <= cur_p):
        return False
    g3 = np.array([gate['x'], gate['y'], gate['z']])
    return float(np.max(np.abs(cur_xyz - g3))) < tol


def _frame_collision(xyz, gates, gate_size, gate_thickness, outer=2.7):
    """Index of the first gate whose solid FRAME the drone is inside, else -1.

    The frame is the band between the inner opening (gate_size/2) and the outer
    extent (outer/2), within +/- gate_thickness/2 of the gate plane. Uses each
    gate's OWN z (fixes the multi-level ladder, where a target-z check misplaces
    the boxes vertically)."""
    half_in, half_out, d = gate_size / 2.0, outer / 2.0, gate_thickness / 2.0
    for i, g in enumerate(gates):
        c, s = math.cos(g['yaw']), math.sin(g['yaw'])
        dx, dy = xyz[0] - g['x'], xyz[1] - g['y']
        nrm = dx * c + dy * s          # along the crossing axis
        lat = -dx * s + dy * c         # lateral (opening width)
        dz = xyz[2] - g['z']           # vertical, THIS gate's z
        if (abs(nrm) < d
                and (abs(lat) > half_in or abs(dz) > half_in)
                and (abs(lat) < half_out and abs(dz) < half_out)):
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--backend', choices=('sim', 'ros'), default='sim')
    ap.add_argument('--uav-yaml', default='config/uav_config_cvar_racing.yaml',
                    help='(sim backend) dynamics config')
    ap.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / 'rl_config.yaml'),
                    help='(ros backend) rl_config.yaml')
    ap.add_argument('--namespace', default='drone0')
    ap.add_argument('--laps', type=int, default=1)
    ap.add_argument('--max-steps-per-gate', type=int, default=1500)
    ap.add_argument('--tol', type=float, default=0.9, help='gate-pass half-size (m)')
    ap.add_argument('--gate-size', type=float, default=0.8, help='inner opening (m); frame begins here')
    ap.add_argument('--gate-thickness', type=float, default=0.5, help='frame thickness along crossing axis (m)')
    ap.add_argument('--strict', action='store_true', help='end the run on a frame collision')
    ap.add_argument('--paced', action='store_true',
                    help='(ros) frozen-physics stepping instead of free-run real-time loop (~58 Hz)')
    ap.add_argument('--motor-perm', default='3,0,1,2',
                    help='their motor i -> our sim motor index')
    ap.add_argument('--step-sleep', type=float, default=None,
                    help='(ros backend) per-step sleep (s); default auto (dt-0.005)')
    ap.add_argument('--viz', action='store_true',
                    help='(sim backend) publish drone + gates to RViz (Fixed Frame: earth)')
    ap.add_argument('--rt-factor', type=float, default=1.0,
                    help='(viz) real-time speed: 1=realtime, 2=2x, 0.5=half')
    ap.add_argument('--body-marker', action='store_true',
                    help='(viz) also draw a sphere body marker (if no RobotModel)')
    ap.add_argument('--stochastic', action='store_true')
    args = ap.parse_args()

    print(f'Loading policy {args.model}')
    model = PPO.load(args.model, device='cpu')
    print('obs_space', model.observation_space.shape,
          'act_space', model.action_space.shape)

    perm = tuple(int(x) for x in args.motor_perm.split(','))
    is_ros = args.backend == 'ros'
    if is_ros:
        from rl.env.quadrace_deploy_env import QuadRaceDeployEnv
        env = QuadRaceDeployEnv(args.config, args.namespace,
                                gates_enu=GATES_ENU, motor_perm=perm,
                                step_sleep=args.step_sleep, free_run=not args.paced)
    else:
        from rl.env.quadrace_sim_env import QuadRaceSimEnv
        env = QuadRaceSimEnv(GATES_ENU, args.uav_yaml, motor_perm=perm)
    print(f'backend: {args.backend}  motor-perm: {perm}')
    n = len(GATES_ENU)
    total_targets = n * args.laps

    bridge = None
    if args.viz and not is_ros:
        from rl.scripts.sim_rviz_bridge import SimRvizBridge
        bridge = SimRvizBridge(base_frame=f'{args.namespace}/base_link',
                               body_marker=args.body_marker)
        viz_sleep = float(getattr(env, '_dt', 0.01)) / max(args.rt_factor, 1e-6)
        print(f'viz: publishing to RViz (Fixed Frame: earth), rt-factor {args.rt_factor}')

    try:
        obs, info = env.reset(start_gate=0)
        prev_xyz = np.asarray(info['start_pos'], dtype=np.float64)
        if is_ros:
            publish_track_markers(env, GATES_ENU)
        if bridge is not None:
            bridge.publish_gates(GATES_ENU, target_idx=env.target_gate_idx)
        passed = 0
        steps_in_gate = 0
        collisions = 0
        colliding = False
        print(f'\nStart on floor behind gate 0; chasing {total_targets} gate(s)')
        t0 = time.time()
        while passed < total_targets:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _r, _term, _trunc, info = env.step(action)
            if is_ros and info['step'] % 100 == 0:
                publish_track_markers(env, GATES_ENU)
            cur_xyz = np.asarray(info['pos_enu'], dtype=np.float64)
            if bridge is not None:
                bridge.publish_drone(cur_xyz, info.get('yaw_enu', 0.0),
                                     quat=info.get('quat_enu'))
                if info['step'] % 20 == 0:
                    bridge.publish_gates(GATES_ENU, target_idx=env.target_gate_idx)
                time.sleep(viz_sleep)
            # frame-collision detection (edge-triggered so we log once per clip)
            hit = _frame_collision(cur_xyz, GATES_ENU, args.gate_size, args.gate_thickness)
            if hit >= 0 and not colliding:
                collisions += 1
                print(f'  !! CLIPPED gate {hit} frame at {cur_xyz.round(2)}')
                if args.strict:
                    print(f'  COLLISION (strict): ending run on gate {hit}')
                    break
            colliding = hit >= 0
            steps_in_gate += 1
            g_idx = passed % n
            if _passed(prev_xyz, cur_xyz, GATES_ENU[g_idx], args.tol):
                passed += 1
                print(f'  PASSED gate {g_idx} (#{passed}/{total_targets}) '
                      f'in {steps_in_gate} steps  pos={cur_xyz.round(2)}')
                steps_in_gate = 0
                if passed >= total_targets:
                    print('  >>> TRACK COMPLETE')
                    break
                if passed % n == 0:   # completed a full lap, rolling into the next
                    print(f'  ===== LAP {passed // n} complete -> starting lap '
                          f'{passed // n + 1}/{args.laps} (continuous) =====')
                env.set_target_gate(passed % n)
                obs = env._get_obs()
                if bridge is not None:
                    bridge.publish_gates(GATES_ENU, target_idx=env.target_gate_idx)
            elif steps_in_gate >= args.max_steps_per_gate:
                print(f'  TIMEOUT on gate {g_idx} after {steps_in_gate} steps  '
                      f'pos={cur_xyz.round(2)}')
                break
            # crude crash/OOB guard
            if cur_xyz[2] < -0.5 or np.abs(cur_xyz[:2]).max() > 60.0:
                print(f'  OOB/crash at {cur_xyz.round(2)} chasing gate {g_idx}')
                break
            prev_xyz = cur_xyz
        dt = time.time() - t0
        print(f'\nDone: {passed}/{total_targets} gates, {collisions} frame collision(s) '
              f'in {dt:.1f}s wall')
    finally:
        env.close()
        if bridge is not None:
            bridge.close()


if __name__ == '__main__':
    main()
