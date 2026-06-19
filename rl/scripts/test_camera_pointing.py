"""Standalone test for the full-3D camera-pointing reward geometry.

Exercises the real `_camera_pointing` helper from the env (no sim needed) and
the pointing-penalty formula from `_observe_step`:

    pointing_norm = cam_off_axis / pi
    penalty       = -k_bearing * pointing_norm   # full strength, no fade

Cases (camera is forward+UP `camera.pitch_deg`, FPV uptilt):
  1. Drone PITCHED toward a level-ahead gate     -> centered  -> penalty ~ 0
  2. Drone ROLLED toward an up-and-to-the-side gate -> centered -> penalty ~ 0
  3. The OPPOSITE attitude for each              -> off-axis  -> big penalty

Each gate is built by pointing the camera optical axis (under the target
attitude) into the world and placing the gate along it, so "centered" is
exact by construction — then we verify the helper agrees and that flipping
the attitude blows the off-axis angle up.

SCOPE — what this offline test does and does NOT prove:
  - DOES: the camera geometry is correct given (roll, pitch, yaw), AND the
    Euler convention used to SET attitude (`_euler_to_quat`, in teleport)
    round-trips exactly through the convention DroneInterface uses to READ
    it (`euler_from_quaternion`). So "teleport to pitch=+30 -> interface
    reads pitch=+30 -> camera math" is fully covered without a sim.
  - Does NOT: prove the running sim / state-estimator republishes the exact
    teleported quaternion (frame handling), nor the PHYSICAL sign (does the
    sim's pitch=+30 actually pitch the nose down in the world). Those need
    the simulator — see `rl/scripts/validate_camera_live.py` (offer) for an
    end-to-end check that teleports to known attitudes and reads back
    `env.drone.orientation` through the real interface.

Run:  python -m rl.scripts.test_camera_pointing
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from as2_python_api.tools.utils import euler_from_quaternion

from rl.env.drone_goal_env import _camera_pointing, _euler_to_quat


def _R_wb(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->world ZYX rotation, same convention as _camera_pointing."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rot_z @ rot_y @ rot_x


def _camera_axis_world(roll: float, pitch: float, yaw: float,
                       cam_pitch: float) -> np.ndarray:
    """World direction the (forward+up) optical axis points under attitude."""
    f_body = np.array([math.cos(cam_pitch), 0.0, math.sin(cam_pitch)])
    return _R_wb(roll, pitch, yaw) @ f_body


def _penalty(off_axis: float, k_bearing: float) -> float:
    pointing_norm = min(off_axis / math.pi, 1.0)
    return -k_bearing * pointing_norm


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(open(here / 'rl_config.yaml'))
    cam_pitch = math.radians(float(cfg['camera']['pitch_deg']))
    k_bearing = float(cfg['reward']['k_bearing'])
    dist = 5.0  # m; gate placement distance (penalty no longer depends on it)

    print(f'camera.pitch_deg={math.degrees(cam_pitch):.0f}  '
          f'k_bearing={k_bearing}  (full-strength pointing penalty, no fade)\n')

    def report(label, rel, roll, pitch, yaw):
        los_cam, off = _camera_pointing(rel, _R_wb(roll, pitch, yaw), cam_pitch)
        az = math.atan2(los_cam[1], los_cam[0])   # from the LOS unit vector
        el = math.atan2(los_cam[2], los_cam[0])
        pen = _penalty(off, k_bearing)
        print(f'  {label:<34s} att(r,p,y)=({math.degrees(roll):+5.0f},'
              f'{math.degrees(pitch):+5.0f},{math.degrees(yaw):+5.0f})deg  '
              f'off_axis={math.degrees(off):6.2f}deg  '
              f'az={math.degrees(az):+6.1f} el={math.degrees(el):+6.1f}  '
              f'penalty={pen:+.4f}')
        return off, pen

    failures = []

    # --- Case 0: Euler set<->read round-trip ----------------------------
    # The teleport sets attitude via _euler_to_quat; DroneInterface reads it
    # back via euler_from_quaternion. They must be exact inverses, else the
    # orientation the reward sees is NOT the one we commanded.
    print('Case 0 — Euler convention round-trip (set via teleport / read via '
          'DroneInterface):')
    for r, p, y in [(0, 0, 0), (30, 0, 0), (0, 30, 0), (20, -15, 60),
                    (-40, 25, -120)]:
        q = _euler_to_quat(math.radians(r), math.radians(p), math.radians(y))
        rr, pp, yy = (math.degrees(a) for a in
                      euler_from_quaternion(q[1], q[2], q[3], q[0]))
        ok = (math.isclose(rr, r, abs_tol=0.1)
              and math.isclose(pp, p, abs_tol=0.1)
              and math.isclose(yy, y, abs_tol=0.1))
        print(f'  set(r,p,y)=({r:+4},{p:+4},{y:+5}) -> '
              f'read=({rr:+6.1f},{pp:+6.1f},{yy:+6.1f})  '
              f'{"OK" if ok else "MISMATCH"}')
        if not ok:
            failures.append(f'euler round-trip ({r},{p},{y})')
    print()

    # --- Case 1: PITCH centers a level-ahead gate -----------------------
    # Up-tilted camera centered on a level-ahead gate requires nose-DOWN
    # (pitch positive). Build the gate along the camera axis at pitch=+cam.
    print('Case 1 — gate level ahead; PITCH down to center:')
    pitch_c = cam_pitch  # nose-down by the camera uptilt -> axis points level-fwd
    rel1 = _camera_axis_world(0.0, pitch_c, 0.0, cam_pitch) * dist
    off, _ = report('pitched toward gate (centered)', rel1, 0.0, pitch_c, 0.0)
    if not math.isclose(off, 0.0, abs_tol=1e-6):
        failures.append('case1 pitched-toward not centered')
    off_lvl, _ = report('  same gate, level (no pitch)', rel1, 0.0, 0.0, 0.0)
    off_opp, _ = report('  same gate, OPPOSITE (nose-up)', rel1, 0.0, -pitch_c, 0.0)
    if not (off_opp > off_lvl > 1e-3):
        failures.append('case1 opposite/level not increasingly off-axis')

    # --- Case 2: ROLL centers an up-and-to-the-side gate ----------------
    print('\nCase 2 — gate up & to the side; ROLL to center:')
    roll_c = math.radians(30.0)
    rel2 = _camera_axis_world(roll_c, 0.0, 0.0, cam_pitch) * dist
    off, _ = report('rolled toward gate (centered)', rel2, roll_c, 0.0, 0.0)
    if not math.isclose(off, 0.0, abs_tol=1e-6):
        failures.append('case2 rolled-toward not centered')
    off_lvl, _ = report('  same gate, level (no roll)', rel2, 0.0, 0.0, 0.0)
    off_opp, _ = report('  same gate, OPPOSITE roll', rel2, -roll_c, 0.0, 0.0)
    if not (off_opp > off_lvl > 1e-3):
        failures.append('case2 opposite/level not increasingly off-axis')

    # --- Case 3: gate behind -> maximal off-axis ------------------------
    print('\nCase 3 — gate behind the camera (worst case):')
    rel3 = _camera_axis_world(0.0, cam_pitch, 0.0, cam_pitch) * (-dist)
    off, pen = report('gate directly behind', rel3, 0.0, cam_pitch, 0.0)
    if not (math.degrees(off) > 170.0):
        failures.append('case3 behind not ~180deg off-axis')

    print()
    if failures:
        print('FAILED:')
        for f in failures:
            print(f'  - {f}')
        raise SystemExit(1)
    print('ALL CHECKS PASSED: pitch and roll both center the gate (penalty ~ 0); '
          'opposite attitudes raise off-axis and the penalty.')


if __name__ == '__main__':
    main()
