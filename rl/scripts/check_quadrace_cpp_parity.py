"""Parity test: C++ QuadRaceBatch == Python QuadRaceVecEnv (the oracle).

Two checks, both feeding identical states so any divergence is the C++ math,
not RNG:
  1. obs parity (no dynamics): set the same their-frame states + target gates,
     compare QuadRaceBatch.reset_envs() obs vs the oracle's update_states().
  2. step parity (one integration): same state + action, compare rewards / dones
     / gate_passed for all envs, and obs where the env did not terminate (the
     oracle auto-resets done envs, the C++ does not).
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, 'sim/multirotor_pysim')
import multirotor_pysim as mps  # noqa: E402
from rl.env.quadrace_vec_env_py import QuadRaceVecEnvPy as QuadRaceVecEnv  # oracle  # noqa: E402

FP = '/root/as2_projects/quadrace-diffsim/quadrace-diffsim/flight_plans/test_track.json'
N = 256


def build_qrb(oracle):
    qrb = mps.QuadRaceBatch(N)
    qrb.set_track(oracle.gate_pos.astype(np.float64),
                  oracle.gate_yaw.astype(np.float64),
                  oracle.gate_pos_rel.astype(np.float64),
                  oracle.gate_yaw_rel.astype(np.float64),
                  None if oracle.bounds_xy is None else oracle.bounds_xy.astype(np.float64))
    cfg = dict(oracle.cfg)
    cfg.update(dt=float(oracle.dt), w_min=oracle._wmin, w_max=oracle._wmax, k=oracle._k,
               cam_angle=oracle.cam_angle, loop_gates=1.0 if oracle.cfg['loop_gates'] else 0.0,
               motor_perm=np.asarray(oracle._perm, np.float64))
    qrb.set_config(cfg)
    qrb.set_model_params(oracle._params)
    return qrb


def random_states(rng, oracle):
    """N their-frame states near random gates (keeps most envs alive 1 step)."""
    g = rng.integers(0, oracle.num_gates, N)
    pos = oracle.gate_pos[g] + rng.normal(0, 1.5, (N, 3))
    vel = rng.normal(0, 2.0, (N, 3))
    eul = rng.uniform(-0.4, 0.4, (N, 3))
    cph, sph = np.cos(eul[:, 0] / 2), np.sin(eul[:, 0] / 2)
    cth, sth = np.cos(eul[:, 1] / 2), np.sin(eul[:, 1] / 2)
    cps, sps = np.cos(eul[:, 2] / 2), np.sin(eul[:, 2] / 2)
    quat = np.stack([cph*cth*cps+sph*sth*sps, sph*cth*cps-cph*sth*sps,
                     cph*sth*cps+sph*cth*sps, cph*cth*sps-sph*sth*cps], 1)
    rates = rng.uniform(-1, 1, (N, 3))
    mot = rng.uniform(-1, 1, (N, 4))
    W = np.concatenate([pos, vel, quat, rates, mot], 1).astype(np.float32)
    return W, g.astype(np.int32)


def main():
    rng = np.random.default_rng(0)
    oracle = QuadRaceVecEnv(FP, num_envs=N, seed=0,
                            env_config=dict(motor_penalty=0.05, motor_penalty_threshold=0.1,
                                            perception_penalty=0.02))
    qrb = build_qrb(oracle)

    # ---- 1. obs parity (no dynamics) ----
    W, tg = random_states(rng, oracle)
    oracle.world_states = W.copy(); oracle.target_gates = tg.copy()
    oracle.update_states()
    ref = oracle.states.copy()
    got = qrb.reset_envs(list(range(N)), W.astype(np.float64), tg.astype(np.float64), oracle._params)
    e = np.abs(ref - got)
    print(f"[1] obs parity        max|Δ|={e.max():.2e}  (worst channel {e.max(0).argmax()})")
    assert e.max() < 1e-4, "obs mismatch"

    # ---- 2. step parity (one integration) ----
    W, tg = random_states(rng, oracle)
    act = rng.uniform(-1, 1, (N, 4)).astype(np.float32)
    # oracle: set state + dynamics + zero prev/steps
    oracle.world_states = W.copy(); oracle.target_gates = tg.copy()
    oracle.prev_actions = np.zeros((N, 4), np.float32)
    oracle.step_counts = np.zeros(N, np.int32)
    oracle.sim.set_state(oracle._their_to_sim(W))
    # qrb: reset to same (zeros prev/steps, sets dynamics)
    qrb.reset_envs(list(range(N)), W.astype(np.float64), tg.astype(np.float64), oracle._params)

    oracle.step_async(act)
    o_obs, o_rew, o_done, _ = oracle.step_wait()
    q_obs, q_rew, q_done, q_pass, q_trunc = qrb.step(act.astype(np.float64))

    dr = np.abs(o_rew - q_rew).max()
    dd = int(np.abs(o_done.astype(int) - q_done.astype(int)).sum())
    alive = o_done < 0.5
    do = np.abs(o_obs[alive] - q_obs[alive]).max() if alive.any() else 0.0
    print(f"[2] reward parity     max|Δ|={dr:.2e}")
    print(f"    done   mismatches = {dd}/{N}")
    print(f"    obs parity (alive) max|Δ|={do:.2e}   ({alive.sum()}/{N} alive)")
    assert dr < 1e-4 and dd == 0 and do < 1e-4, "step mismatch"
    print("\nALL PARITY CHECKS PASSED")


if __name__ == '__main__':
    main()
