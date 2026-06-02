"""SB3 callback that freezes simulator physics during PPO weight updates.

SB3 training loop:
    on_rollout_start  →  collect rollout  →  on_rollout_end  →  train()  →  repeat

Physics is paused in on_rollout_end (before train()) and resumed in
on_rollout_start (before the next rollout), so drones stay still while
gradients are computed.
"""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from stable_baselines3.common.callbacks import BaseCallback
from std_srvs.srv import SetBool


class PhysicsPauseCallback(BaseCallback):
    """Pause/resume simulator physics around the PPO update step."""

    def __init__(self, namespaces: list[str], verbose: int = 0):
        super().__init__(verbose)
        self._namespaces = namespaces
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._exec_thread: threading.Thread | None = None
        self._clients: dict[str, rclpy.client.Client] = {}

    def _init_callback(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node('rl_physics_pause')
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._exec_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._exec_thread.start()

        for ns in self._namespaces:
            client = self._node.create_client(SetBool, f'/{ns}/pause_physics')
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    f'/{ns}/pause_physics service unavailable after 10s. '
                    'Make sure the patched simulator is running.')
            self._clients[ns] = client

    def _set_physics(self, pause: bool) -> None:
        req = SetBool.Request()
        req.data = pause
        futures = [c.call_async(req) for c in self._clients.values()]
        deadline = time.monotonic() + 2.0
        while not all(f.done() for f in futures) and time.monotonic() < deadline:
            time.sleep(0.005)

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        self._set_physics(True)

    def on_rollout_start(self) -> None:
        self._set_physics(False)

    def on_training_end(self) -> None:
        self._set_physics(False)
        if self._executor:
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()
