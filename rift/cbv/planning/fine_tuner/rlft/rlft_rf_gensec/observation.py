"""PLUTO features plus the frozen RF-Gensec three-vehicle safety state."""
import numpy as np
import carla
from rift.gym_carla.observation.pluto_observation import CBVPlutoObservation
from rift.gym_carla.reward.cbv_reward import CBVDenseReward
from rift.gym_carla.utils.common import get_relative_info
from rift.scenario.tools.carla_data_provider import CarlaDataProvider as CDP
from rift.scenario.tools.timer import GameTime


def penalized_collision(event, ego_id, actor_type):
    return bool(event and event.get('other_actor_id') != ego_id
                and actor_type.startswith('vehicle.'))


class RIFTEnvironmentReward(CBVDenseReward):
    def __init__(self, planner):
        super().__init__(planner)
        self.ego_vehicle = None
        self.angular_history = {}

    def set_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.angular_history.clear()

    def get_reward(self, collisions):
        result = {}
        actors = CDP.get_CBVs_by_ego(self.ego_vehicle.id)
        for actor_id, actor in actors.items():
            event = collisions.get(actor_id)
            other = CDP.get_world().get_actor(event['other_actor_id']) if event else None
            collision = penalized_collision(event, self.ego_vehicle.id,
                                            event.get('other_actor_type', other.type_id if other else '')
                                            if event else '')
            delta_dis, delta_angle = self.get_delta_info(actor)
            speed = CDP.get_velocity(actor).length()
            acc = CDP.get_acceleration(actor).length()
            omega = np.deg2rad(actor.get_angular_velocity().z)
            now = GameTime.get_time()
            previous = self.angular_history.get(actor_id)
            angular_acc = ((omega - previous[1]) / (now - previous[0])
                           if previous and now > previous[0] else 0.0)
            self.angular_history[actor_id] = (now, omega)
            offroad = not bool(self.map.get_waypoint(
                CDP.get_location(actor), project_to_road=False,
                lane_type=carla.LaneType.Driving))
            result[actor_id] = self.reward_model.get_reward(
                delta_dis, delta_angle, speed, acc, omega, angular_acc,
                int(collision), int(offroad))
        self.angular_history = {k: v for k, v in self.angular_history.items() if k in actors}
        return result


class RFGensecPlutoObservation(CBVPlutoObservation):
    type = 'cbv_rlft_rf_gensec_obs'

    def __init__(self, env_params, CBVAction,
                 config_path='rift/cbv/planning/config/rlft_rf_gensec.yaml'):
        super().__init__(env_params, CBVAction, config_path)
        self.CBV_reward = RIFTEnvironmentReward(self.CBV_route_planner)

    def _get_obs(self, observations, actor_id, actor, ego):
        super()._get_obs(observations, actor_id, actor, ego)
        transform = CDP.get_transform(actor)
        matrix = np.array(transform.get_matrix())
        yaw = np.deg2rad(transform.rotation.yaw)
        vehicles = [actor, ego]
        vehicles.extend(a for a in CDP.get_CBV_nearby_agents(ego.id, actor_id)
                        if a.id not in (actor_id, ego.id))
        state = np.zeros((3, 6), dtype=np.float32)
        for row, vehicle in enumerate(vehicles[:3]):
            state[row] = get_relative_info(vehicle, yaw, matrix)
        observations[actor_id]['rf_state'] = state
