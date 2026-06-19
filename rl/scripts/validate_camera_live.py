"""End-to-end validation of the camera-pointing geometry against the LIVE
simulator + DroneInterface (NOT pure math).

Requires the patched Aerostack2 sim running for the target drone, e.g.:
    ./launch_as2.bash            # single drone (drone0), or
    ./launch_as2.bash -m -d 0    # swarm; run this on drone0's domain

The gate stays fixed at the env's target; the DRONE is flown to several
positions/attitudes *relative to* the gate (never on top of it), and at each
pose we hold attitude and print the LIVE camera off-axis angle + the exact
pointing reward term training uses, repeatedly, so you can read it before the
pose changes.

What it actually checks, by driving the real env:

  A. ORIENTATION READBACK — teleport the drone to known (roll, pitch, yaw)
     and read `env.drone.orientation` back through the real ROS pose
     pipeline. Confirms the sim/state-estimator reports the same attitude
     we commanded (frame handling end-to-end), not just that two helper
     functions invert each other.

  B. PHYSICAL SIGN (the real arbiter) — teleport to a pure pitch, then a
     pure roll, with hover thrust + zero rates, unpause one short burst, and
     read the resulting WORLD-frame velocity. Tilt makes the thrust vector
     push the drone in the tilt direction, so the velocity sign tells us the
     true physical meaning of +pitch and +roll:
        +pitch should push +x (forward)  -> "pitch positive = nose-DOWN"
        +roll  should push -y (right)     -> "roll positive  = RIGHT"
     If either is reversed, the corresponding sign in `_camera_pointing`
     (and the obs/reward) is wrong for this stack.

  C. CAMERA OFF-AXIS, LIVE, FROM SEVERAL POSITIONS — for each scenario the
     drone is placed at a real offset from the gate and held while the live
     off-axis angle + pointing reward are printed every `--watch-dt` seconds
     for `--watch` seconds. "Centered" scenarios put the gate on the live
     camera axis (expect off_axis ~ 0, penalty ~ 0); "off" scenarios use a
     wrong attitude (expect large off_axis / penalty).

Run:  python -m rl.scripts.validate_camera_live --namespace drone0
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from rl.env.drone_goal_env import DroneGoalEnv


def _R_wb(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rot_z @ rot_y @ rot_x


def _camera_axis_world(roll, pitch, yaw, cam_pitch) -> np.ndarray:
    """World direction the forward+up optical axis points under attitude."""
    f_body = np.array([math.cos(cam_pitch), 0.0, math.sin(cam_pitch)])
    return _R_wb(roll, pitch, yaw) @ f_body


def _pointing_penalty(env: DroneGoalEnv, s) -> float:
    """Replicate the LIVE pointing reward term from `_observe_step`:
        penalty = -k_bearing * (cam_off_axis/pi)
    (full strength, no distance fade)."""
    k_bearing = float(env.cfg['reward']['k_bearing'])
    pointing_norm = min(s.cam_off_axis / math.pi, 1.0)
    return -k_bearing * pointing_norm


def _settle_at(env: DroneGoalEnv, pos, roll, pitch, yaw, settle_s):
    """Teleport to (pos, roll, pitch, yaw), hold attitude (zero rates + hover
    thrust) for settle_s of unpaused sim, then pause and let the final pose
    propagate to the DroneInterface. Returns the live _compute_state()."""
    env._teleport(pos, yaw, roll=roll, pitch=pitch)
    env._set_physics(False)
    env._send_rates_command(0.0, 0.0, 0.0, env._thrust_hover)
    time.sleep(settle_s)
    env._set_physics(True)
    time.sleep(0.05)  # let the executor thread deliver the last pose
    return env._compute_state()


def _watch(env: DroneGoalEnv, expect_deg: float,
           watch_s: float, dt: float) -> tuple[float, float]:
    """With physics paused, print the LIVE off-axis + pointing reward
    repeatedly for watch_s so it can be read before the pose changes.
    Returns (first_off_axis_deg, first_penalty)."""
    d2 = math.degrees
    first_off = None
    first_pen = None
    n = max(1, int(round(watch_s / dt)))
    for i in range(n):
        s = env._compute_state()
        off = d2(s.cam_off_axis)
        pen = _pointing_penalty(env, s)
        dist = float(np.linalg.norm(np.asarray(s.rel_world, dtype=np.float64)))
        if first_off is None:
            first_off, first_pen = off, pen
        remaining = watch_s - i * dt
        print(f'    [{remaining:4.1f}s] off_axis={off:6.2f}deg '
              f'(expect ~{expect_deg:4.1f})  az={d2(s.cam_azimuth):+6.1f} '
              f'el={d2(s.cam_elevation):+6.1f}  dist={dist:4.1f}m '
              f'penalty={pen:+.4f}', flush=True)
        time.sleep(dt)
    return first_off, first_pen


def _drift_burst(env: DroneGoalEnv, pos, roll, pitch, yaw, burst_s=0.3):
    """Teleport to a tilted attitude with zero velocity, then let it
    accelerate under hover thrust for burst_s. Returns the live world-frame
    velocity (used to read the physical tilt direction)."""
    env._teleport(pos, yaw, roll=roll, pitch=pitch,
                  lin_vel=np.zeros(3), ang_vel=np.zeros(3))
    env._set_physics(False)
    env._send_rates_command(0.0, 0.0, 0.0, env._thrust_hover)
    time.sleep(burst_s)
    env._set_physics(True)
    time.sleep(0.05)
    return env._read_velocity()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--namespace', type=str, default='drone0')
    parser.add_argument('--tol-att-deg', type=float, default=6.0,
                        help='attitude readback tolerance (deg)')
    parser.add_argument('--tol-off-deg', type=float, default=8.0,
                        help='camera off-axis tolerance (deg)')
    parser.add_argument('--settle', type=float, default=0.4,
                        help='seconds to hold attitude before reading the pose')
    parser.add_argument('--watch', type=float, default=4.0,
                        help='seconds to display the live reward per pose')
    parser.add_argument('--watch-dt', type=float, default=0.5,
                        help='seconds between live reward prints')
    args = parser.parse_args()

    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config) if args.config else here / 'rl_config.yaml'

    print(f'Connecting to sim as {args.namespace} ...')
    env = DroneGoalEnv(config_path=str(cfg_path), drone_namespace=args.namespace)
    if env._action_mode != 'rates':
        print('WARNING: action.mode is not "rates"; this validation targets '
              'rates mode (the camera obs/reward path).')
    cam_pitch = env._camera_pitch
    gate = np.asarray(env._target_pos, dtype=np.float64).copy()
    # A generic spawn well away from the gate (and from the OOB walls) for the
    # attitude/dynamics checks — never on top of the gate.
    home = gate + np.array([-min(5.0, env._oob_x_ext * 0.6), 0.0, 0.0])
    d2 = math.degrees
    failures: list[str] = []

    try:
        # ----- A. Orientation readback ---------------------------------
        print('\n=== A. Orientation readback (commanded -> live DroneInterface) ===')
        for r, p, y in [(0, 0, 0), (30, 0, 0), (0, 30, 0), (0, 0, 45),
                        (20, -15, 60)]:
            s = _settle_at(env, home, math.radians(r), math.radians(p),
                           math.radians(y), args.settle)
            rr, pp, yy = d2(s.roll), d2(s.pitch), d2(s.yaw)
            ok = (abs(rr - r) <= args.tol_att_deg
                  and abs(pp - p) <= args.tol_att_deg
                  and abs(((yy - y + 180) % 360) - 180) <= args.tol_att_deg)
            print(f'  cmd(r,p,y)=({r:+4},{p:+4},{y:+5})  '
                  f'live=({rr:+6.1f},{pp:+6.1f},{yy:+6.1f})  '
                  f'{"OK" if ok else "MISMATCH"}')
            if not ok:
                failures.append(f'A: readback ({r},{p},{y})')

        # ----- B. Physical sign via dynamics ---------------------------
        print('\n=== B. Physical sign (tilt -> which way does it accelerate) ===')
        v = _drift_burst(env, home, 0.0, math.radians(20), 0.0)
        nose = 'forward(+x)' if v[0] > 0 else 'backward(-x)'
        concl = 'nose-DOWN' if v[0] > 0 else 'nose-UP'
        print(f'  pitch=+20deg -> world vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) '
              f'-> moves {nose} -> +pitch means {concl}')
        if v[0] <= 0:
            failures.append('B: +pitch is not nose-down (camera pitch sign '
                            'in _camera_pointing must be revisited)')
        v = _drift_burst(env, home, math.radians(20), 0.0, 0.0)
        side = 'right(-y)' if v[1] < 0 else 'left(+y)'
        concl = 'RIGHT' if v[1] < 0 else 'LEFT'
        print(f'  roll=+20deg  -> world vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) '
              f'-> moves {side} -> +roll means {concl}')
        if v[1] >= 0:
            failures.append('B: +roll is not to the right (roll sign '
                            'assumption is reversed)')

        # ----- C. Camera off-axis, LIVE, from several positions --------
        print('\n=== C. Camera off-axis (live reward held per pose) ===')

        def scenario(label, roll_c, pitch_c, yaw_c, dist, place):
            """place='centered': drone offset so the gate sits on its live
            camera axis at `dist` m (predicted off_axis ~ 0).
            place='off': drone placed `dist` m straight behind the gate (-x)
            holding the given attitude, so the camera misses the gate.

            For BOTH, the expected off-axis is predicted analytically from the
            commanded attitude (camera axis) vs. the drone->gate line-of-sight,
            and the LIVE env reading is checked against that prediction."""
            rc, pc, yc = (math.radians(roll_c), math.radians(pitch_c),
                          math.radians(yaw_c))
            axis = _camera_axis_world(rc, pc, yc, cam_pitch)
            if place == 'centered':
                pos = gate - axis * dist
            else:  # 'off' — sit behind the gate, attitude won't aim at it
                pos = gate + np.array([-dist, 0.0, 0.0])
            los = gate - pos
            los = los / max(1e-9, float(np.linalg.norm(los)))
            exp_deg = d2(math.acos(max(-1.0, min(1.0, float(np.dot(axis, los))))))
            print(f'\n  {label}')
            print(f'    drone at ({pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f})  '
                  f'gate at ({gate[0]:+.1f},{gate[1]:+.1f},{gate[2]:+.1f})  '
                  f'att(r,p,y)=({roll_c:+.0f},{pitch_c:+.0f},{yaw_c:+.0f})  '
                  f'predicted off_axis={exp_deg:.1f}deg')
            _settle_at(env, pos, rc, pc, yc, args.settle)
            off, _ = _watch(env, exp_deg, args.watch, args.watch_dt)
            ok = abs(off - exp_deg) <= args.tol_off_deg
            if not ok:
                failures.append(f'C: {label} (live={off:.1f}, '
                                f'predicted={exp_deg:.1f})')
            print(f'    -> {"OK" if ok else "MISMATCH"}')

        # Centered from a spread of positions/attitudes (different relative
        # geometry each time; gate never under the drone):
        scenario('centered, level, gate ahead+up', 0, 0, 0, 5.0, 'centered')
        scenario('centered, pitched nose-down', 0, d2(cam_pitch), 0, 5.0,
                 'centered')
        scenario('centered, rolled right', 30, 0, 0, 5.0, 'centered')
        scenario('centered, yawed + close', 0, 0, 35, 3.0, 'centered')
        scenario('centered, far away', 0, 0, 0, 7.0, 'centered')
        # Off-axis controls (gate behind, level / wrong attitude):
        scenario('OFF: behind gate, level', 0, 0, 0, 5.0, 'off')
        scenario('OFF: behind gate, nose-up', 0, -25, 0, 5.0, 'off')

        print('\n' + ('FAILED:\n  ' + '\n  '.join(failures) if failures
                      else 'ALL LIVE CHECKS PASSED — orientation readback, '
                           'physical sign, and camera off-axis all consistent '
                           'with the env code.'))
        if failures:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == '__main__':
    main()
