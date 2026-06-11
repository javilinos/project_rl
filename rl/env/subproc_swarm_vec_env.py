"""SubprocVecEnv backend: one DroneGoalEnv per subprocess for true parallelism.

The default ``SwarmDummyVecEnv`` runs all drones in one Python process; every
per-env ROS service call, ``DroneInterface`` attribute read, and ``time.sleep``
is serialized under one GIL. With N=10 drones at 50 Hz the GIL-bound overhead
caps the effective control rate at ~10 Hz wall clock.

``SubprocSwarmVecEnv`` instead launches one subprocess per drone. Each
subprocess has its own Python interpreter, its own rclpy context, and its own
``DroneInterfaceTeleop`` executor thread. ``env.step`` (including the
``time.sleep(dt)``) runs in parallel across N processes, so total wall-clock
cost per step is ``dt + max_individual_overhead`` instead of
``dt + sum_of_overheads``.

The ``PhysicsPauseCallback`` lives in the main process but dispatches the
pause/resume via ``VecEnv.env_method`` so each subprocess calls its own
``/sim_clock_publisher/pause_physics`` service on its own ROS_DOMAIN_ID.
That's necessary with per-drone DDS isolation: the main process can't
reach any drone's domain directly, so each subprocess has to fire the
pause locally — and each drone has its own ``sim_clock_publisher_node``
on its own domain anyway, because /clock is not shared across domains.
"""

from __future__ import annotations

import os

from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.env.drone_goal_env import DroneGoalEnv


def _factory(config_path: str, namespace: str, publish_gate_marker: bool,
             ros_domain_id: int | None):
    """Build a top-level (pickle-able) env factory for one subprocess."""
    def _fn():
        # Set ROS_DOMAIN_ID before rclpy.init() (called inside DroneGoalEnv).
        # With start_method='spawn' each subprocess starts fresh, so this
        # env-var change is local to the subprocess and matches the platform
        # processes launched by `launch_as2.bash -d <base>`.
        if ros_domain_id is not None:
            os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
        return DroneGoalEnv(
            config_path=config_path,
            drone_namespace=namespace,
            publish_gate_marker=publish_gate_marker,
        )
    return _fn


def make_subproc_swarm_vec_env(
    config_path: str,
    namespaces: list[str],
    start_method: str = 'spawn',
    base_ros_domain_id: int | None = None,
) -> SubprocVecEnv:
    """One DroneGoalEnv per subprocess; true process-level parallelism.

    ``start_method='spawn'`` is required: fork would inherit the parent's
    rclpy context and DDS participant handles, which is unsafe with threads
    and can deadlock the simulator services. spawn gives each subprocess a
    clean interpreter that calls ``rclpy.init()`` itself.

    Every subprocess publishes its own per-drone gate Marker on
    ``/rl/target_gate``. The markers share the topic but use unique
    ``Marker.id`` (derived from the drone namespace), so RViz tracks them
    independently — N drones with N randomized targets show up as N gates.

    If ``base_ros_domain_id`` is set, each subprocess sets ``ROS_DOMAIN_ID``
    to ``base + drone_index`` before initialising rclpy — matching the IDs
    that ``launch_as2.bash -d <base>`` assigned to each drone's platform
    processes. This isolates each drone's DDS traffic.
    """
    if not namespaces:
        raise ValueError('namespaces must contain at least one drone id.')

    fns = [
        _factory(
            config_path, ns, publish_gate_marker=True,
            ros_domain_id=(base_ros_domain_id + i
                           if base_ros_domain_id is not None else None),
        )
        for i, ns in enumerate(namespaces)
    ]
    print(f'creating SubprocSwarmVecEnv with namespaces: {namespaces} '
          f'(start_method={start_method}, base_ros_domain_id={base_ros_domain_id})')
    return SubprocVecEnv(fns, start_method=start_method)
