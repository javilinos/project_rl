"""Random-action smoke test for DroneGoalEnv (single drone or swarm).

Usage:
    # Single drone (default).
    python -m rl.scripts.smoke --steps 200

    # Swarm of 3 drones (requires ./launch_as2.bash -m).
    python -m rl.scripts.smoke --swarm --steps 200

    # Specific namespaces.
    python -m rl.scripts.smoke --namespaces drone0,drone1 --steps 200

    # Teleport-state regression tests.
    python -m rl.scripts.smoke --state-test
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import yaml

from rl.env.drone_goal_env import DroneGoalEnv
from rl.env.swarm_vec_env import make_swarm_vec_env


def _smoke_single(cfg_path: Path, steps: int) -> None:
    env = DroneGoalEnv(config_path=str(cfg_path))
    try:
        obs, info = env.reset(seed=0)
        print('initial obs:', obs, 'target:', info['target_pos'])
        episode_reward = 0.0
        episodes = 0
        for i in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                print(
                    f'[step {i}] reward={episode_reward:.2f} '
                    f'reason={ {k: v for k, v in info.items() if k in ("success", "oob", "timeout")} } '
                    f'dist_xy={info.get("dist_xy", float("nan")):.3f} '
                    f'bearing={info.get("bearing", float("nan")):.3f}'
                )
                episodes += 1
                obs, info = env.reset(seed=i + 1)
                episode_reward = 0.0
        print(f'completed {steps} steps across {episodes} finished episodes')
    finally:
        env.close()


def _smoke_swarm(cfg_path: Path, namespaces: list[str], steps: int) -> None:
    vec = make_swarm_vec_env(config_path=str(cfg_path), namespaces=namespaces)
    try:
        obs = vec.reset()
        print(f'initial vec obs shape={obs.shape}')
        ep_rewards = np.zeros(vec.num_envs, dtype=np.float64)
        ep_done_counts = np.zeros(vec.num_envs, dtype=np.int64)
        for i in range(steps):
            actions = np.stack([vec.action_space.sample() for _ in range(vec.num_envs)])
            obs, rewards, dones, infos = vec.step(actions)
            ep_rewards += rewards
            for j, done in enumerate(dones):
                if done:
                    info = infos[j]
                    reason = {k: v for k, v in info.items() if k in ('success', 'oob', 'TimeLimit.truncated')}
                    print(
                        f'[step {i}] env={j} ({namespaces[j]}) ep_reward={ep_rewards[j]:.2f} '
                        f'reason={reason} dist_xy={info.get("dist_xy", float("nan")):.3f} '
                        f'bearing={info.get("bearing", float("nan")):.3f}'
                    )
                    ep_rewards[j] = 0.0
                    ep_done_counts[j] += 1
        print(f'completed {steps} swarm steps; episodes finished per env: {ep_done_counts.tolist()}')
    finally:
        vec.close()


def _check_observations(env: DroneGoalEnv, target_pos: np.ndarray) -> tuple[int, int]:
    """Verify _get_obs() returns the body-frame state.

    obs[0..2]: body-frame relative position to target (xy / max_dist_xy, z / pos_max_z)
    obs[3..5]: body-frame velocity tanh-normalized by v_max (should be ≈0 after settling)
    obs[6]:    bearing-to-target = atan2(rel_body_y, rel_body_x) divided by π.
               0 → target directly ahead; ±1 → target directly behind;
               +0.5 → target to the left; −0.5 → target to the right.
               (Previously this slot held drone_yaw − target_yaw; that
               changed when the reward stopped paying for gate-yaw
               alignment — see _get_obs docstring.)
    obs[7..8]: only in action.mode=rates — roll/(π/2), pitch/(π/2)
    """
    max_dist_xy = env._max_dist_xy
    pos_max_z = env._pos_max_z
    target_yaw = env._target_yaw

    def wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    # (description, drone_pos, drone_yaw)
    cases: list[tuple[str, np.ndarray, float]] = [
        ('at target, aligned',        np.array([0.0,  0.0, 1.75]), target_yaw),
        ('+x +y offset',              np.array([1.0,  2.0, 1.5]),  target_yaw),
        ('-x -y offset',              np.array([-1.5, -1.0, 2.0]), target_yaw),
        ('z offset (low)',            np.array([0.0,  0.0, 1.0]),  target_yaw),
        ('z offset (high)',           np.array([0.0,  0.0, 2.5]),  target_yaw),
        ('yaw_err=+π/2',              np.array([0.0,  0.0, 1.75]), target_yaw + math.pi / 2),
        ('yaw_err=-π/2',              np.array([0.0,  0.0, 1.75]), target_yaw - math.pi / 2),
        ('yaw_err≈+π (wraparound)',   np.array([0.0,  0.0, 1.75]), target_yaw + math.pi - 0.05),
    ]

    obs_tol = 0.03
    n_pass = 0
    for desc, drone_pos, drone_yaw in cases:
        env._teleport(drone_pos, drone_yaw)
        env._send_speed_command(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.2)

        obs = env._get_obs()
        rel = target_pos - drone_pos
        cos_y = math.cos(drone_yaw)
        sin_y = math.sin(drone_yaw)
        rel_body_x = cos_y * rel[0] + sin_y * rel[1]
        rel_body_y = -sin_y * rel[0] + cos_y * rel[1]
        yaw_rel = wrap(drone_yaw - target_yaw)
        # Velocity after teleport+settle should be ~0 (physics paused).
        expected = np.array([
            np.clip(rel_body_x / max_dist_xy, -1.0, 1.0),
            np.clip(rel_body_y / max_dist_xy, -1.0, 1.0),
            np.clip(rel[2] / pos_max_z, -1.0, 1.0),
            0.0, 0.0, 0.0,
            yaw_rel / math.pi,
        ], dtype=np.float32)
        err = np.abs(obs - expected)
        ok = bool(np.all(err < obs_tol))
        n_pass += int(ok)
        print(
            f'[{" OK " if ok else "FAIL"}] {desc:<28s} '
            f'expected={[round(float(v), 3) for v in expected]} '
            f'got={[round(float(v), 3) for v in obs]} '
            f'max_err={err.max():.4f}'
        )
    return n_pass, len(cases)


def _check_reward_formula(env: DroneGoalEnv) -> tuple[int, int]:
    """Verify the per-step reward formula:
        r = -k_pos·dist_xy_norm
            - k_height·z_err_norm
            - k_bearing·bearing_norm
            - k_yaw_align·yaw_err_norm
            + k_speed·horizontal_speed_score
            - k_vertical·vertical_speed_norm
    All inputs in [0, 1].
    """
    rw = env.cfg['reward']
    k_pos = float(rw['k_pos'])
    k_height = float(rw['k_height'])
    k_bearing = float(rw['k_bearing'])
    k_yaw_align = float(rw['k_yaw_align'])
    k_speed = float(rw['k_speed'])
    k_vertical = float(rw['k_vertical'])

    # (desc, d, z, b, y, h_speed, v_speed, expected)
    cases: list[tuple[str, float, float, float, float, float, float, float]] = [
        ('zero state',              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ('dist_xy_norm=1 only',     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -k_pos),
        ('z_err_norm=1 only',       0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -k_height),
        ('bearing_norm=1 only',     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -k_bearing),
        ('yaw_err_norm=1 only',     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -k_yaw_align),
        ('forward_speed=1 only',    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, +k_speed),
        ('vertical_speed=1 only',   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -k_vertical),
        ('mixed all six',           0.4, 0.3, 0.25, 0.5, 0.6, 0.2,
            -k_pos * 0.4 - k_height * 0.3 - k_bearing * 0.25 - k_yaw_align * 0.5
            + k_speed * 0.6 - k_vertical * 0.2),
    ]

    n_pass = 0
    for desc, d, z, b, y, h, v, expected in cases:
        actual = env._compute_reward(d, z, b, y, h, v)
        ok = abs(actual - expected) < 1e-5
        n_pass += int(ok)
        print(
            f'[{" OK " if ok else "FAIL"}] {desc:<28s} '
            f'expected={expected:+.5f} got={actual:+.5f} '
            f'diff={actual - expected:+.2e}'
        )
    return n_pass, len(cases)


def _check_termination(env: DroneGoalEnv,
                       target_pos: np.ndarray) -> tuple[int, int]:
    """Verify gate-frame termination logic:
    - success: |depth| < depth_tol AND max(|lat|, |vert|) < inner_half
    - crash (oob): |depth| < depth_tol AND inner_half ≤ max(|lat|, |vert|) < outer_half
    Drone is teleported with zero velocity so the terminal speed factor is 0,
    but `info['terminal_bonus']` must still be ≥ 0 and the success/oob/crash
    flags must match expectations.
    """
    target_yaw = env._target_yaw
    gate = env.cfg['gate']
    inner_half = float(gate['size_interior']) / 2.0  # 0.75
    outer_half = float(gate['size_exterior']) / 2.0  # 1.35

    # (description, drone_pos offset (world), drone_yaw, expect_success, expect_crash)
    cases: list[tuple[str, np.ndarray, float, bool, bool]] = [
        ('at gate centre',            np.array([0.0,  0.0, 0.0]), target_yaw, True,  False),
        ('inner offset (lateral)',    np.array([0.0,  0.5, 0.0]), target_yaw, True,  False),
        ('inner offset (vertical)',   np.array([0.0,  0.0, 0.5]), target_yaw, True,  False),
        ('frame crash (lateral)',     np.array([0.0,  0.9, 0.0]), target_yaw, False, True),
        ('frame crash (vertical)',    np.array([0.0,  0.0, 1.0]), target_yaw, False, True),
        ('far in front of gate',      np.array([1.5,  0.0, 0.0]), target_yaw, False, False),
        ('past the gate (depth>tol)', np.array([-1.5, 0.0, 0.0]), target_yaw, False, False),
        ('outside frame (lateral)',   np.array([0.0,  1.5, 0.0]), target_yaw, False, False),
    ]

    n_pass = 0
    for desc, offset, drone_yaw, expect_success, expect_crash in cases:
        drone_pos = target_pos + offset
        env._step_idx = 0
        env._last_action = np.zeros(4, dtype=np.float32)
        env._teleport(drone_pos, drone_yaw)
        env._send_speed_command(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.2)

        _, reward, terminated, _, info = env._observe_step()
        got_success = bool(info.get('success', False))
        got_crash = bool(info.get('crash', False))
        expect_terminated = expect_success or expect_crash

        success_ok = got_success == expect_success
        crash_ok = got_crash == expect_crash
        terminated_ok = bool(terminated) == expect_terminated
        if expect_success:
            fields_ok = all(
                k in info for k in
                ('align_factor', 'speed_factor', 'center_factor', 'terminal_bonus'))
            bonus_ok = info.get('terminal_bonus', -1.0) >= 0.0
        else:
            fields_ok = True
            bonus_ok = True

        ok = success_ok and crash_ok and terminated_ok and fields_ok and bonus_ok
        n_pass += int(ok)
        outcome = ('success' if got_success
                   else 'crash' if got_crash
                   else 'continue')
        expected = ('success' if expect_success
                    else 'crash' if expect_crash
                    else 'continue')
        print(
            f'[{" OK " if ok else "FAIL"}] {desc:<28s} '
            f'got={outcome:<8s} expected={expected:<8s} '
            f"depth={info.get('gate_depth', float('nan')):+.2f} "
            f"lat={info.get('gate_lateral', float('nan')):+.2f} "
            f"vert={info.get('gate_vertical', float('nan')):+.2f}"
        )
    _ = inner_half, outer_half  # silence unused vars (kept for clarity)
    return n_pass, len(cases)


def _check_reset_on_success(env: DroneGoalEnv,
                            target_pos: np.ndarray) -> tuple[int, int]:
    """Verify that reaching the goal produces terminated=True, and that a
    subsequent reset() places the drone at a fresh init pose with a clean
    step counter."""
    min_dist = float(env.cfg.get('min_init_target_dist', 0.0))

    # Force a success state: pin target, teleport drone on top of it,
    # pre-advance the step counter so we can confirm reset() zeroes it.
    env._target_pos = target_pos.copy()
    env._step_idx = 5
    env._teleport(target_pos, 0.0)
    env._send_speed_command(0.0, 0.0, 0.0, 0.0)
    time.sleep(0.2)

    _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    pos_at_goal, _ = env._read_pose()

    # Now call reset and inspect the new episode state.
    _, info_after = env.reset(seed=0)
    pos_after, _ = env._read_pose()
    target_after = info_after['target_pos']
    target_yaw_after = info_after['target_yaw']
    config_target_yaw = float(env.cfg['target'].get('yaw', 0.0))

    checks = [
        ('success flag set on goal',
            bool(info.get('success', False))),
        ('terminated=True on goal',
            bool(terminated)),
        ('reset moved drone off goal',
            float(np.linalg.norm(pos_after - pos_at_goal)) >= min_dist - 0.1),
        ('reset cleared step_idx',
            env._step_idx == 0),
        ('new init pose not at goal',
            float(np.linalg.norm(pos_after - target_after)) >= min_dist - 0.1),
        ('target pos still at config-fixed point',
            bool(np.allclose(target_after, target_pos, atol=1e-3))),
        ('target yaw fixed at config value',
            abs(target_yaw_after - config_target_yaw) < 1e-6),
    ]

    n_pass = 0
    for desc, ok in checks:
        n_pass += int(ok)
        print(f'[{" OK " if ok else "FAIL"}] {desc}')
    return n_pass, len(checks)


def _gate_side_flight(env: DroneGoalEnv, target_pos: np.ndarray,
                      start_offset: np.ndarray, drone_yaw: float,
                      action: np.ndarray, label: str,
                      max_steps: int = 100) -> None:
    """Fly the drone from a known start with a fixed action; stream the
    reward + state breakdown at every step until termination.
    """
    rw = env.cfg['reward']
    k_pos = float(rw['k_pos'])
    k_height = float(rw['k_height'])
    k_bearing = float(rw['k_bearing'])
    k_yaw_align = float(rw['k_yaw_align'])
    k_speed = float(rw['k_speed'])
    k_vertical = float(rw['k_vertical'])
    oob_pen = float(rw['oob_penalty'])
    crash_pen = float(rw.get('crash_penalty', 0.0))

    env._target_pos = target_pos.copy()
    env._target_yaw = 0.0
    env._step_idx = 0
    env._last_action = np.zeros(4, dtype=np.float32)
    env._teleport(target_pos + start_offset, drone_yaw)
    env._send_speed_command(0.0, 0.0, 0.0, 0.0)
    env._publish_gate_marker()   # refresh RViz to match this run
    env._set_physics(True)
    time.sleep(0.2)

    print(f'\n--- {label} ---')
    print(f'  start = target + {start_offset.tolist()}, '
          f'drone_yaw={drone_yaw:+.2f}, action={action.tolist()}')
    print('  step | depth   n·vel  dist_xy | r_pos    r_hgt    r_brg    '
          'r_yaw    r_spd    r_vrt  | r_cont   r_term  r_total | flags')

    total_reward = 0.0
    for step in range(max_steps):
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Reconstruct continuous vs terminal split.
        terminal_bonus = float(info.get('terminal_bonus', 0.0))
        terminal_pen = 0.0
        if info.get('crash'):
            terminal_pen = -crash_pen
        elif info.get('oob'):
            terminal_pen = -oob_pen
        r_cont = reward - terminal_bonus - terminal_pen
        r_term = terminal_bonus + terminal_pen

        # Each continuous reward component reconstructed from the normalized
        # values the env stored in info. Sum should equal r_cont.
        r_pos = -k_pos * float(info.get('dist_xy_norm', 0.0))
        r_hgt = -k_height * float(info.get('z_err_norm', 0.0))
        r_brg = -k_bearing * float(info.get('bearing_norm', 0.0))
        r_yaw = -k_yaw_align * float(info.get('yaw_err_norm', 0.0))
        r_spd = +k_speed * float(info.get('forward_speed_score', 0.0))
        r_vrt = -k_vertical * float(info.get('vertical_speed_norm', 0.0))

        # Signed gate_normal·vel from the drone's real world velocity.
        vel_world = env._read_velocity()
        gate_normal_dot_vel = (
            math.cos(env._target_yaw) * vel_world[0]
            + math.sin(env._target_yaw) * vel_world[1]
        )

        flags = []
        if info.get('success'):
            flags.append('SUCCESS')
        if info.get('crash'):
            flags.append('CRASH')
        if info.get('back_entry'):
            flags.append('back_entry')
        if info.get('oob') and not info.get('crash'):
            flags.append('OOB')
        if truncated:
            flags.append('timeout')

        depth = float(info.get('gate_depth', float('nan')))
        dist_xy = float(info.get('dist_xy', float('nan')))
        print(
            f'  [{step:3d}] {depth:+6.3f} {gate_normal_dot_vel:+6.3f} '
            f'{dist_xy:6.3f} | '
            f'{r_pos:+7.4f} {r_hgt:+7.4f} {r_brg:+7.4f} '
            f'{r_yaw:+7.4f} {r_spd:+7.4f} {r_vrt:+7.4f} | '
            f'{r_cont:+7.4f} {r_term:+7.2f} {reward:+7.3f} | '
            f"{' '.join(flags)}"
        )

        if terminated or truncated:
            break

    print(f'  episode total reward = {total_reward:+.3f}')


def _check_gate_entry_side(env: DroneGoalEnv, target_pos: np.ndarray) -> None:
    """Probe the gate entrance: fly the drone from several start poses with
    a fixed action and stream the reward breakdown so you can see at which
    point success or crash fires, and how the continuous reward evolves.

    With ``target_yaw = 0`` the gate normal is +x. Legit forward crossing
    means world velocity has a +x component; legit approach side is -x.
    """
    print('\n=== Gate entry side flight probe ===')
    print(f'  target_pos = {target_pos.tolist()}, target_yaw = 0.0  '
          '→ legit cross = +x, legit approach side has gate_depth > 0')

    # Scenario A: -x side, command body-forward = world +x. Should reach the
    # gate plane with positive n·vel → SUCCESS.
    _gate_side_flight(
        env, target_pos,
        start_offset=np.array([-3.0, 0.0, 0.0]),
        drone_yaw=0.0,
        action=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        label='A: legit approach (-x side, body +forward)',
    )

    # Scenario B: +x side, command body-forward = world +x. Body-forward goes
    # AWAY from the gate; gate_depth grows, drone should hit OOB eventually
    # without ever crossing.
    _gate_side_flight(
        env, target_pos,
        start_offset=np.array([3.0, 0.0, 0.0]),
        drone_yaw=0.0,
        action=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        label='B: wrong side, body +forward = away from gate',
    )

    # Scenario C: +x side, drone faces -x (yaw=π) and commands body-forward.
    # That sends it toward the gate, but in the -x world direction →
    # gate_normal·vel < 0 → CRASH back_entry when it reaches the opening.
    _gate_side_flight(
        env, target_pos,
        start_offset=np.array([3.0, 0.0, 0.0]),
        drone_yaw=math.pi,
        action=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        label='C: back-entry (drone yaw=π, body +forward = world -x)',
    )

    # Leave drone parked on the legit approach side for the user to inspect
    # the gate's decoration in RViz.
    legit_pos = target_pos + np.array([-3.0, 0.0, 0.0])
    env._teleport(legit_pos, 0.0)
    env._send_speed_command(0.0, 0.0, 0.0, 0.0)
    env._publish_gate_marker()
    print(f'\n  drone parked at {legit_pos.tolist()} facing +x — look at RViz:')
    print('    the gate decoration should be on the -x side facing the drone.')
    print('    if it is on the +x side, the +π in _publish_gate_marker is '
          'flipping it the wrong way.')
    time.sleep(3.0)


def _smoke_state_test(cfg_path: Path) -> None:
    """Run observation, reward-formula, termination/yaw-bonus, and
    reset-on-success checks against a single DroneGoalEnv instance.
    """
    env = DroneGoalEnv(config_path=str(cfg_path))
    target_pos = np.array([
        float(env.cfg['target']['x']),
        float(env.cfg['target']['y']),
        float(env.cfg['target']['z']),
    ], dtype=np.float64)
    env._target_pos = target_pos

    try:
        print('=== observation tests ===')
        p_obs, t_obs = _check_observations(env, target_pos)
        print('\n=== reward formula tests ===')
        p_rew, t_rew = _check_reward_formula(env)
        print('\n=== termination tests ===')
        p_term, t_term = _check_termination(env, target_pos)
        print('\n=== reset on success tests ===')
        p_res, t_res = _check_reset_on_success(env, target_pos)

        # Diagnostic-only probe (no pass/fail counts) so the user can both
        # read the synthetic outcomes and look at RViz at the same time.
        _check_gate_entry_side(env, target_pos)

        total_pass = p_obs + p_rew + p_term + p_res
        total = t_obs + t_rew + t_term + t_res
        print(f'\noverall: {total_pass}/{total} passed '
              f'(obs {p_obs}/{t_obs}, reward {p_rew}/{t_rew}, '
              f'term {p_term}/{t_term}, reset {p_res}/{t_res})')
        if total_pass < total:
            raise SystemExit(1)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--swarm', action='store_true',
                        help='Use SwarmDummyVecEnv with namespaces from config.')
    parser.add_argument('--namespaces', type=str, default=None,
                        help='Comma-separated drone namespaces. Implies --swarm.')
    parser.add_argument('--state-test', action='store_true',
                        help='Run teleport-state regression tests and exit.')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config) if args.config else here / 'rl_config.yaml'

    if args.state_test:
        _smoke_state_test(cfg_path)
    elif args.namespaces:
        namespaces = [ns.strip() for ns in args.namespaces.split(',') if ns.strip()]
        _smoke_swarm(cfg_path, namespaces, args.steps)
    elif args.swarm:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        drone_cfg = cfg.get('drone', {})
        namespaces = drone_cfg.get('namespaces') or [drone_cfg.get('namespace', 'drone0')]
        _smoke_swarm(cfg_path, namespaces, args.steps)
    else:
        _smoke_single(cfg_path, args.steps)


if __name__ == '__main__':
    main()
