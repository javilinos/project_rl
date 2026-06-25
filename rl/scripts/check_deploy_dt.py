"""Measure the ACTUAL sim-time advanced per QuadRaceDeployEnv.step().

The deploy steps by unpausing the sim clock, sleeping dt=0.01 wall, then
pausing. Because the pause is a service round-trip, the real integrated sim-dt
is 0.01 + latency (and jittery), unlike their JAX training's exact 0.01. This
script subscribes to /clock and reports the per-step sim-dt distribution so we
know whether it's worth making the step deterministic.

Run (sim up): python -m rl.scripts.check_deploy_dt
"""

from __future__ import annotations

import numpy as np
from rosgraph_msgs.msg import Clock

from rl.env.quadrace_deploy_env import QuadRaceDeployEnv
from rl.scripts.run_quadrace_policy import GATES_ENU
from pathlib import Path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--step-sleep', type=float, default=None,
                    help='override per-step sleep (s); default auto (dt-0.005)')
    args = ap.parse_args()
    cfg = str(Path(__file__).resolve().parent.parent / 'rl_config.yaml')
    env = QuadRaceDeployEnv(cfg, 'drone0', gates_enu=GATES_ENU,
                            step_sleep=args.step_sleep)
    print(f'step_sleep = {env._step_sleep*1e3:.2f} ms')

    latest = {'t': None}
    env._svc_node.create_subscription(
        Clock, '/clock',
        lambda m: latest.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9),
        10)

    def sim_t():
        return latest['t']

    try:
        env.reset(start_gate=0)
        # hover-ish constant action so the drone stays controllable while we time
        a = np.full(4, -0.3, dtype=np.float32)
        dts = []
        prev = sim_t()
        for _ in range(200):
            env.step(a)
            t = sim_t()
            if prev is not None and t is not None:
                dts.append(t - prev)
            prev = t
        d = np.array(dts[5:])  # drop warmup
        print(f'target dt = 0.0100 s   (control_hz={1.0/env._dt:.0f})')
        print(f'measured sim-dt/step over {len(d)} steps:')
        print(f'  mean={d.mean()*1e3:.3f} ms   std={d.std()*1e3:.3f} ms')
        print(f'  min ={d.min()*1e3:.3f} ms   max={d.max()*1e3:.3f} ms')
        print(f'  inflation vs 0.01: {(d.mean()/0.01 - 1)*100:+.1f}%')
    finally:
        env.close()


if __name__ == '__main__':
    main()
