"""Build per-env ModelParams arrays for multirotor_pysim.BatchSim from our
uav_config YAML, with optional domain randomization.

The keys match BatchSim.set_model_params. Single-env deploy uses nominal
(dr=None); training passes a dr dict {field: percent} to randomize per env
(uniform value*(1 +/- percent/100)).
"""

from __future__ import annotations

import numpy as np
import yaml

# Default racing config (matches config/uav_config_cvar_racing.yaml).
DEFAULT_UAV_YAML = "config/uav_config_cvar_racing.yaml"


def _model_block(uav_yaml: str) -> dict:
    with open(uav_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    # ROS param file: /**: { ros__parameters: { multirotor: {...} } }
    root = next(iter(cfg.values()))["ros__parameters"]
    return root["multirotor"]["dynamics"]["model"]


def build_params(uav_yaml: str = DEFAULT_UAV_YAML, num_envs: int = 1,
                 dr: dict | None = None, seed: int | None = None) -> dict:
    m = _model_block(uav_yaml)
    mp = m["motors_params"]
    n = int(num_envs)
    rng = np.random.default_rng(seed)

    def rand(nominal, key, shape):
        """Tile nominal to (n, *shape); if dr[key] given, randomize +/- percent."""
        base = np.broadcast_to(np.asarray(nominal, float), shape).copy()
        out = np.tile(base, (n,) + (1,) * len(shape))
        if dr and key in dr and dr[key]:
            lo, hi = 1.0 - dr[key] / 100.0, 1.0 + dr[key] / 100.0
            out = out * rng.uniform(lo, hi, size=out.shape)
        return out

    # Motor geometry: explicit asymmetric layout if present, else quad-X.
    if all(k in mp for k in ("motors_x", "motors_y", "motors_direction")):
        mx = np.asarray(mp["motors_x"], float)
        my = np.asarray(mp["motors_y"], float)
        md = np.asarray(mp["motors_direction"], float)
    else:
        xd, yd = float(mp["x_dist"]), float(mp["y_dist"])
        mx = np.array([xd, xd, -xd, -xd])
        my = np.array([-yd, yd, yd, -yd])
        md = np.array([1.0, -1.0, 1.0, -1.0])

    params = dict(
        mass=rand(m["vehicle_mass"], "mass", ()),
        inertia=rand(m["vehicle_inertia"], "inertia", (3,)),
        drag=rand(m.get("vehicle_drag_coefficient", 0.0), "drag", ()),
        rotor_drag=rand(m.get("rotor_drag_coefficient", 0.0), "rotor_drag", ()),
        body_quad=rand(m.get("body_quadratic_drag", [0, 0, 0]), "body_quad", (3,)),
        thrust_k_angle=rand(m.get("thrust_k_angle", 0.0), "thrust_k_angle", ()),
        thrust_k_hor=rand(m.get("thrust_k_hor", 0.0), "thrust_k_hor", ()),
        thrust_aero_radius=rand(m.get("thrust_aero_radius", 0.0), "thrust_aero_radius", ()),
        aero_moment=rand(m.get("vehicle_aero_moment_coefficient", [0, 0, 0]), "aero_moment", (3,)),
        thrust_coeff=rand(mp["thrust_coefficient"], "thrust_coeff", ()),
        torque_coeff=rand(mp["torque_coefficient"], "torque_coeff", ()),
        min_speed=rand(mp["min_speed"], "min_speed", ()),
        max_speed=rand(mp["max_speed"], "max_speed", ()),
        time_constant=rand(mp["time_constant"], "time_constant", ()),
        rotational_inertia=rand(mp["rotational_inertia"], "rotational_inertia", ()),
        motors_x=rand(mx, "motors_x", (4,)),
        motors_y=rand(my, "motors_y", (4,)),
        motors_direction=np.tile(md, (n, 1)),  # never randomize spin direction
    )
    return params
