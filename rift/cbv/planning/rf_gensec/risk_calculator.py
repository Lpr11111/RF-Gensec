"""Spatiotemporal vehicle risk field used by RF-Gensec.

The observation convention is ``[..., actor, feature]`` with features
``[x, y, length, width, relative_yaw, forward_speed]``.  Actor zero is the
coordinate-system centre; the final route row can be excluded with
``actor_rows``.

Lateral speed decay is deliberately absent. The field's decay coefficients are

    delta_x = beta_x * max(|x| - L/2, 0) / (alpha_x * v_x + 1)
    delta_y = beta_y * max(|y| - W/2, 0)

because the observation's sixth feature is a scalar *forward* speed
(``get_relative_info`` in ``rift/gym_carla/utils/common.py``); no lateral
velocity component ``v_y`` is available. Adding an ``alpha_y`` term would
require extending the observation and re-collecting the offline HJR dataset, so
the paper states the lateral decay without a speed term rather than claiming a
symmetric implementation.

Coordinate conventions differ across the three entry points and must not be
"unified" without re-deriving the frames:

``calculate_dynamic_risk``
    R_CBV. Each *other* actor is the field source and the centre CBV is the
    measurement point. The observation stores actors in the CBV frame, so each
    source's field must be rotated into that source's own body frame first;
    hence the explicit inverse rotation by ``-relative_yaw``.

``calculate_center_risk_at_actor``
    R_AV. The centre CBV is the field source and another actor is the
    measurement point. No rotation is applied, and none is needed: the
    observation frame *is* the CBV body frame (row 0 has zero position and zero
    relative yaw by construction), so the target's stored ``(x, y)`` is already
    expressed in the source's body frame.

``calculate_vehicle_risk``
    World-frame helper for risk-field plotting. Takes absolute locations and
    performs its own rotation into the source vehicle's body frame.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class RiskCalculator:
    def __init__(self, alpha_x: float = 0.631,
                 beta_x: float = 1.0, beta_y: float = 1.0):
        self.alpha_x = alpha_x
        self.beta_x = beta_x
        self.beta_y = beta_y

    def calculate_dynamic_risk(self, obs_data: np.ndarray,
                               actor_rows: Optional[int] = None) -> np.ndarray:
        obs = np.asarray(obs_data, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[None, ...]
        if obs.ndim != 3 or obs.shape[-1] != 6:
            raise ValueError("obs_data must have shape (N, actors, 6) or (actors, 6)")

        actors = obs[:, :actor_rows] if actor_rows is not None else obs
        risks = np.zeros(actors.shape[0], dtype=np.float32)
        for batch_index, vehicles in enumerate(actors):
            max_risk = 0.0
            for vehicle in vehicles[1:]:
                x_rel, y_rel, length, width, yaw_rel, speed = vehicle
                if np.allclose(vehicle, 0.0) or np.hypot(x_rel, y_rel) > 100.0:
                    continue
                cos_yaw, sin_yaw = np.cos(-yaw_rel), np.sin(-yaw_rel)
                x_center = -(x_rel * cos_yaw - y_rel * sin_yaw)
                y_center = -(x_rel * sin_yaw + y_rel * cos_yaw)
                delta_x = self.beta_x * max(abs(x_center) - 0.5 * length, 0.0) / (
                    self.alpha_x * abs(speed) + 1.0)
                delta_y = self.beta_y * max(abs(y_center) - 0.5 * width, 0.0)
                max_risk = max(max_risk, 1.0 / (np.hypot(delta_x, delta_y) + 1.0))
            risks[batch_index] = max_risk
        return risks

    def calculate_center_risk_at_actor(self, obs_data: np.ndarray,
                                       actor_index: int = 1) -> np.ndarray:
        """Evaluate the centre CBV's risk field at another observed actor."""
        obs = np.asarray(obs_data, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[None, ...]
        if obs.ndim != 3 or obs.shape[-1] != 6:
            raise ValueError("obs_data must have shape (N, actors, 6) or (actors, 6)")
        if actor_index <= 0 or actor_index >= obs.shape[1]:
            raise ValueError("actor_index must identify a non-centre actor")

        values = np.zeros(obs.shape[0], dtype=np.float32)
        for index, (source, target) in enumerate(zip(obs[:, 0], obs[:, actor_index])):
            if np.allclose(target, 0.0):
                continue
            x_rel, y_rel = target[:2]
            length, width, _, speed = source[2:]
            delta_x = self.beta_x * max(abs(x_rel) - 0.5 * length, 0.0) / (
                self.alpha_x * abs(speed) + 1.0)
            delta_y = self.beta_y * max(abs(y_rel) - 0.5 * width, 0.0)
            values[index] = 1.0 / (np.hypot(delta_x, delta_y) + 1.0)
        return values

    def calculate_static_risk(self, obs_data: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(obs_data).shape[0], dtype=np.float32)

    def calculate_traffic_light_risk(self, obs_data: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(obs_data).shape[0], dtype=np.float32)

    def calculate_total_risk(self, obs_data: np.ndarray,
                             actor_rows: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        dynamic = self.calculate_dynamic_risk(obs_data, actor_rows=actor_rows)
        static = np.zeros_like(dynamic)
        traffic_light = np.zeros_like(dynamic)
        return dynamic + static + traffic_light, {
            "dynamic_risk": dynamic,
            "static_risk": static,
            "traffic_light_risk": traffic_light,
        }

    def calculate_vehicle_risk(self, vehicle_state: dict, ego_state: dict) -> float:
        """Calculate one vehicle's field at the ego position.

        State dictionaries use FREA's ``location``, ``velocity``, ``yaw`` and
        ``extent`` convention. Yaw is expected in radians.
        """
        vehicle_location = np.asarray(vehicle_state["location"], dtype=np.float32)
        ego_location = np.asarray(ego_state["location"], dtype=np.float32)
        velocity = np.asarray(vehicle_state["velocity"], dtype=np.float32)
        length, width = np.asarray(vehicle_state["extent"], dtype=np.float32)[:2]
        relative = ego_location[:2] - vehicle_location[:2]
        yaw = -float(vehicle_state["yaw"])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        x_center = relative[0] * cos_yaw - relative[1] * sin_yaw
        y_center = relative[0] * sin_yaw + relative[1] * cos_yaw
        speed = np.linalg.norm(velocity[:2])
        delta_x = self.beta_x * max(abs(x_center) - 0.5 * length, 0.0) / (
            self.alpha_x * speed + 1.0)
        delta_y = self.beta_y * max(abs(y_center) - 0.5 * width, 0.0)
        return float(1.0 / (np.hypot(delta_x, delta_y) + 1.0))
