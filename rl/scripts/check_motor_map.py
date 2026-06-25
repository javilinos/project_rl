"""Empirically find the motor-index correspondence between OUR multirotor sim
and THEIR MonoRace mixer, so the policy's 4 motor commands hit the right rotors.

Their mixer (quad_race/environment.py, FRD body frame), torque contribution of
each motor as its speed rises:
    Mx(roll)  = -k_p1 W1^2 - k_p2 W2^2 + k_p3 W3^2 + k_p4 W4^2
    My(pitch) = -k_q1 W1^2 + k_q2 W2^2 - k_q3 W3^2 + k_q4 W4^2
    Mz(yaw)   = -k_r1 W1  + k_r2 W2  + k_r3 W3  - k_r4 W4
=> per-motor (roll, pitch, yaw) sign signature, THEIR (FRD) frame:
    motor 1: (-,-,-)   motor 2: (-,+,+)   motor 3: (+,-,+)   motor 4: (+,+,-)

Method: hover-base all 4 motors, pulse ONE, integrate a few ms from rest, read
our body rates omega (FLU). Convert to THEIR frame (p, -q, -r) and read the
sign. The (roll, pitch) pair uniquely identifies which of their motors our
motor plays; yaw then tells us if prop spin directions agree.

Run (sim up): python -m rl.scripts.check_motor_map
Then pass the printed permutation to run_quadrace_policy --motor-perm a,b,c,d
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rl.env.quadrace_deploy_env import QuadRaceDeployEnv
from rl.scripts.run_quadrace_policy import GATES_ENU

# their per-motor (roll, pitch, yaw) signs, FRD frame
THEIR_SIG = {
    0: (-1, -1, -1),  # their motor 1
    1: (-1, +1, +1),  # their motor 2
    2: (+1, -1, +1),  # their motor 3
    3: (+1, +1, -1),  # their motor 4
}

BASE = 650.0     # rad/s baseline on all motors
DELTA = 350.0    # rad/s pulse on the probed motor
INTEGRATE = 0.05  # s to let the torque build measurable omega
TRIALS = 3


def probe(env, j):
    """Average (roll, pitch, yaw) body-rate response (THEIR frame) to pulsing
    OUR motor j."""
    acc = np.zeros(3)
    for _ in range(TRIALS):
        env._teleport(np.array([0.0, 0.0, 4.0]), 0.0)   # level hover, at rest
        env._send_motor_command([BASE] * 4)
        env._set_physics(True)
        time.sleep(0.05)
        cmd = [BASE] * 4
        cmd[j] += DELTA
        env._send_motor_command(cmd)
        env._set_physics(False)
        time.sleep(INTEGRATE)
        env._set_physics(True)
        w = env._read_omega()                # FLU body rates
        acc += np.array([w[0], -w[1], -w[2]])  # -> THEIR FRD frame
    return acc / TRIALS


def main():
    cfg = str(Path(__file__).resolve().parent.parent / 'rl_config.yaml')
    env = QuadRaceDeployEnv(cfg, 'drone0', gates_enu=GATES_ENU)
    try:
        print('Probing our 4 motors (pulse one, read body-rate response)...\n')
        sigs = {}
        for j in range(4):
            r = probe(env, j)
            s = tuple(int(np.sign(v)) if abs(v) > 1e-3 else 0 for v in r)
            sigs[j] = s
            print(f'  our motor {j}: response(roll,pitch,yaw)={r.round(3)} '
                  f'-> sign {s}')

        # match each our-motor to their motor by (roll, pitch)
        perm = [None] * 4   # perm[their_i] = our_j
        yaw_ok = True
        for j, s in sigs.items():
            match = [i for i, ts in THEIR_SIG.items()
                     if ts[0] == s[0] and ts[1] == s[1]]
            if len(match) == 1:
                i = match[0]
                perm[i] = j
                if s[2] != 0 and s[2] != THEIR_SIG[i][2]:
                    yaw_ok = False
            else:
                print(f'  !! our motor {j} sign {s} did not uniquely match '
                      f'(roll,pitch) — check BASE/DELTA or sim state')

        print('\n--- RESULT ---')
        if all(p is not None for p in perm):
            print(f'motor-perm (their i -> our j): {perm}')
            print(f'run with:  --motor-perm {",".join(str(p) for p in perm)}')
        else:
            print(f'incomplete perm: {perm}  (some motors unmatched)')
        if not yaw_ok:
            print('WARNING: yaw (spin) signs DISAGREE with their mixer at the '
                  'matched positions. A permutation cannot fix inverted yaw — '
                  'our sim prop spin directions are mirrored vs theirs. Flip the '
                  'CW/CCW pattern in config/uav_config_cvar_racing.yaml '
                  '(motors_direction) so yaw torque matches, then re-probe.')
        else:
            print('yaw (spin) signs agree — permutation alone should fix it.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
