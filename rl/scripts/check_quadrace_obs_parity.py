"""Offline validation of rl/env/quadrace_adapter.py against the REAL
QuadRace.update_states (no Aerostack2 sim, no policy needed).

Three checks:
  1. compute_relative_gates  == env.gate_pos_rel / gate_yaw_rel
  2. assemble_obs_their_frame == env.update_states() obs, over random states
  3. full ENU->their pipeline sanity: a level drone sitting at a gate, facing
     the crossing direction, yields ~zero pos/vel/euler in the obs.

Run: python -m rl.scripts.check_quadrace_obs_parity
(expects their repo at /root/as2_projects/quadrace-diffsim/quadrace-diffsim with
options.json pointing at the test_track flight plan).
"""

from __future__ import annotations

import sys
import numpy as np

THEIR_REPO = "/root/as2_projects/quadrace-diffsim/quadrace-diffsim"
sys.path.insert(0, THEIR_REPO)
import os
os.chdir(THEIR_REPO)  # so options.json / flight_plans resolve

from quad_race.environment import QuadRace, get_quat_from_euler  # noqa: E402

sys.path.insert(0, "/root/as2_projects/project_rl")
from rl.env.quadrace_adapter import (  # noqa: E402
    assemble_obs_their_frame, compute_relative_gates,
    vec_enu_to_their, yaw_enu_to_their, rotmat_enu_to_their,
    body_rates_enu_to_their, rotmat_to_quat, motor_speed_to_norm,
)


def main() -> None:
    rng = np.random.default_rng(0)
    env = QuadRace(num_envs=8)
    n = env.num_gates
    print(f"track gates: {n} | loop_gates: {env.loop_gates}")

    # ---- Check 1: relative-gate table -------------------------------------
    pos_rel, yaw_rel = compute_relative_gates(env.gate_pos, env.gate_yaw)
    e1 = np.abs(pos_rel - env.gate_pos_rel).max()
    e1y = np.abs(yaw_rel - env.gate_yaw_rel).max()
    print(f"[1] relative-gate table   max|dpos|={e1:.2e}  max|dyaw|={e1y:.2e}")
    assert e1 < 1e-4 and e1y < 1e-4, "relative-gate mismatch"

    # ---- Check 2: obs assembly vs update_states ---------------------------
    worst = 0.0
    for trial in range(200):
        tgt = int(rng.integers(0, n))
        # random valid their-frame world state
        gp = env.gate_pos[tgt]
        pos = gp + rng.normal(0, 3, 3)
        vel = rng.normal(0, 5, 3)
        eul = rng.uniform(-np.pi, np.pi, 3)
        quat = np.array(get_quat_from_euler(eul.reshape(3, 1))).reshape(4)
        rates = rng.uniform(-3, 3, 3)
        mw = rng.uniform(-1, 1, 4)
        state = np.concatenate([pos, vel, quat, rates, mw]).astype(np.float32)

        env.world_states[0] = state
        env.target_gates[0] = tgt
        env.update_states()
        ref = env.states[0].copy()

        nxt = (tgt + 1) % n if env.loop_gates else min(tgt + 1, n - 1)
        mine = assemble_obs_their_frame(
            state[0:3], state[3:6], state[6:10], state[10:13], state[13:17],
            env.gate_pos[tgt % n], env.gate_yaw[tgt % n],
            env.gate_pos_rel[nxt], env.gate_yaw_rel[nxt],
        )
        d = np.abs(mine - ref)
        if d.max() > worst:
            worst = float(d.max())
            worst_idx = int(d.argmax())
    print(f"[2] obs assembly parity   worst |mine-ref| = {worst:.3e} "
          f"(channel {worst_idx}) over 200 random states")
    assert worst < 1e-3, f"obs assembly mismatch {worst}"

    # ---- Check 3: full ENU->their physical sanity -------------------------
    # ENU: drone level, sitting exactly at gate 0's opening centre, facing the
    # gate's crossing direction. Expect pos_G~0, vel_G~0, euler~0, psi-gate~0.
    g_their = env.gate_pos[0]
    g_yaw_their = float(env.gate_yaw[0])
    # invert Rx(180) to get the ENU equivalents the live sim would report
    pos_enu = vec_enu_to_their(g_their)          # T is self-inverse
    yaw_enu = yaw_enu_to_their(g_yaw_their)
    vel_enu = np.zeros(3)
    # level FLU drone yawed to yaw_enu: R_wb = Rz(yaw_enu)
    cy, sy = np.cos(yaw_enu), np.sin(yaw_enu)
    r_wb_enu = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])

    pos_t = vec_enu_to_their(pos_enu)
    vel_t = vec_enu_to_their(vel_enu)
    quat_t = rotmat_to_quat(rotmat_enu_to_their(r_wb_enu))
    omega_t = body_rates_enu_to_their(np.zeros(3))
    mw_norm = motor_speed_to_norm(np.full(4, 510.0))
    nxt = 1 % n
    obs = assemble_obs_their_frame(
        pos_t, vel_t, quat_t, omega_t, mw_norm,
        env.gate_pos[0], env.gate_yaw[0], env.gate_pos_rel[nxt], env.gate_yaw_rel[nxt])
    print(f"[3] ENU->their sanity     pos_G={obs[0:3]}  vel_G={obs[3:6]}  "
          f"euler={obs[6:9]}")
    assert np.abs(obs[0:3]).max() < 1e-4, "pos_G should be ~0 at gate"
    assert np.abs(obs[6:9]).max() < 1e-4, "euler/yaw-rel should be ~0 facing gate"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
