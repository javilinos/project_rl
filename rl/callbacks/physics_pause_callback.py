"""SB3 callback that freezes simulator physics during PPO weight updates.

SB3 training loop:
    on_rollout_start  →  collect rollout  →  on_rollout_end  →  train()  →  repeat

The callback dispatches the pause/resume to each env via ``VecEnv.env_method``
so it doesn't need its own ROS node. That keeps it domain-agnostic — each
subprocess (potentially on its own ROS_DOMAIN_ID) calls its own ``_set_physics``
on its own rclpy context, which is the only one that can reach its drone's
``/sim_clock_publisher/pause_physics`` service (one clock publisher per
DDS domain — see ``rl/env/subproc_swarm_vec_env.py``).
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class PhysicsPauseCallback(BaseCallback):
    """Pause/resume simulator physics around the PPO update step."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        # step_wait already paused everything before observing; this is a
        # no-op on the platform side but keeps the local cache flag consistent.
        self.training_env.env_method('_set_physics', True)

    def on_rollout_start(self) -> None:
        # Resume every drone before the next rollout begins. Each subprocess
        # calls its own _set_physics on its own ROS context.
        self.training_env.env_method('_set_physics', False)

    def on_training_end(self) -> None:
        # Best-effort resume so the simulator isn't left frozen after a run.
        try:
            self.training_env.env_method('_set_physics', False)
        except Exception:
            pass
