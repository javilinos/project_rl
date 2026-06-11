"""Vectorized DummyVecEnv that ticks all drones in parallel within one process.

Plain DummyVecEnv calls ``env.step`` sequentially, so the per-step ``time.sleep
(dt)`` in DroneGoalEnv runs once per env, multiplying wall-clock cost by the
number of drones. ``SwarmDummyVecEnv`` instead:

1. Broadcasts the velocity command to every drone (no sleep).
2. Sleeps once for one control period.
3. Reads observation / reward / termination from every drone.

Each env still owns its own ROS node, so simulator timestepping, state
publishing, and target tracking happen in parallel during the shared sleep.
"""

from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env.drone_goal_env import DroneGoalEnv


class SwarmDummyVecEnv(DummyVecEnv):
    """DummyVecEnv specialization that batches the per-step sleep.

    Constructor takes a list of zero-arg env factories like the parent class.
    All envs must share the same control period (``cfg.control_hz``).
    """

    def __init__(self, env_fns):
        super().__init__(env_fns)
        # Sanity: confirm every env exposes the two-phase API.
        for env in self.envs:
            if not hasattr(env, '_apply_action') or not hasattr(env, '_observe_step'):
                raise TypeError(
                    'SwarmDummyVecEnv requires envs with _apply_action / '
                    '_observe_step (e.g. DroneGoalEnv).'
                )
        self._dt = float(self.envs[0]._dt)
        # Detect heterogeneous dt early — the shared sleep would be wrong.
        for env in self.envs[1:]:
            if abs(env._dt - self._dt) > 1e-9:
                raise ValueError(
                    'All envs in a SwarmDummyVecEnv must share the same '
                    'control_hz.'
                )
        self._send_time: float | None = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        # _apply_action unpauses each env's physics individually.
        for env, action in zip(self.envs, actions):
            env._apply_action(action)
        self._send_time = time.monotonic()

    def step_wait(self):
        # Sleep one control period of *real* time so the simulator integrates
        # exactly dt with the new command. Then freeze each drone's physics
        # *before* reading its state so the observation reflects the moment
        # of pause, not a state that keeps drifting while we read it. The
        # freeze also persists through the outer loop (policy.predict,
        # bookkeeping…) so the next dt window starts cleanly.
        if self._send_time is not None:
            elapsed = time.monotonic() - self._send_time
            if elapsed < self._dt:
                time.sleep(self._dt - elapsed)

        for env_idx in range(self.num_envs):
            env = self.envs[env_idx]
            env._set_physics(True)
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = \
                env._observe_step()
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]['TimeLimit.truncated'] = truncated and not terminated
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs, self.reset_infos[env_idx] = env.reset()
            self._save_obs(env_idx, obs)

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )


def make_swarm_vec_env(config_path: str, namespaces: list[str]) -> SwarmDummyVecEnv:
    """Build a ``SwarmDummyVecEnv`` with one ``DroneGoalEnv`` per namespace."""
    if not namespaces:
        raise ValueError('namespaces must contain at least one drone id.')

    def _factory(ns: str):
        def _fn():
            return DroneGoalEnv(config_path=config_path, drone_namespace=ns)
        return _fn
    
    print(f'creating SwarmDummyVecEnv with namespaces: {namespaces}')

    return SwarmDummyVecEnv([_factory(ns) for ns in namespaces])
