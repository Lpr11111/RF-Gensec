# Adapted for RF-Gensec: retain method infrastructure and remove comparison-policy behavior.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : pluto.py
@Date    : 2024/10/24
'''
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import numpy.typing as npt

from typing import Any, Dict, List, Tuple

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.cbv.planning.pluto.model.pluto_model import PlanningModel
from rift.cbv.planning.pluto.controller.pid_controller import PIDController
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.util.logger import Logger
from rift.util.torch_util import CUDA, get_device_name


class TrajectoryBackbone(CBVBasePolicy):
    name = 'rf_gensec_backbone'
    type = 'il'

    def __init__(self, config, logger: Logger):
        super().__init__(config, logger)
        self.logger = logger
        self.radius = config['obs']['radius']
        self.num_scenario = config['num_scenario']

        self._use_prediction = config['use_prediction']
        self._topk = config['topk']
        self._ckpt_path = config['ckpt_path']
        
        self._frame_rate = config['frame_rate']  # default 10 fps
        self._step_interval = 1.0 / self._frame_rate  # default 0.1s
        self.device = get_device_name()
        
        # the pluto model for inference
        self.pluto_model = CUDA(PlanningModel(radius=self.radius))

        # the PID controller dict (each CBV need a unique PID controller)
        self.controllers = defaultdict(lambda: defaultdict(lambda: PIDController(sample_interval=self._frame_rate)))

        # render
        self._render = config['need_video_render']
        if self._render:
            self.reset_render_data()

    def reset_render_data(self):
        self._render_data = {env_id: {
            "ego_states": {},
            "nearby_agents_states": {},
            "CBV_states": {},
            "route_ids_list": [],
            "reference_lines_list": [],
            "route_waypoints_list": [],
            "interaction_wp_list": [],
            "planning_trajectory_list": [],
            "candidate_trajectories_list": [],
            "candidate_index_list": [],
            "predictions_list": [],
        } for env_id in range(self.num_scenario)}

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            raise ValueError(f'Pluto policy not support training mode.')
        elif mode == 'eval':
            self.pluto_model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')



    
    def _clean_CBVs(self, infos, CBVs_obs_list):
        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            env_id = info['env_id']
            active_ids = set(CBVs_obs.keys())
            if env_id in self.controllers:
                # remove unused CBVs and keep default controller
                for cbv_id in list(self.controllers[env_id].keys()):
                    if cbv_id not in active_ids:
                        del self.controllers[env_id][cbv_id]
                
                if not self.controllers[env_id]:
                    del self.controllers[env_id]

    def get_render_data(self, env_id):
        return self._render_data[env_id]

    def load_infer_checkpoint(self, checkpoint: str, device_name: str):
        ckpt = torch.load(checkpoint, map_location=device_name)
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        return state_dict
    
    def load_train_checkpoint(self, checkpoint: str, device_name: str):
        ckpt = torch.load(checkpoint, map_location=device_name)
        return ckpt['state_dict']

    def load_model(self, resume=True):
        # load the model
        self.pluto_model.load_state_dict(self.load_infer_checkpoint(self._ckpt_path, device_name=self.device))
    
    def save_model(self, episode):
        raise NotImplementedError()
    
    

    def get_control(self, env_id, CBV_id, center_state: CarlaAgentState, local_trajectory: np.ndarray) -> Tuple[float, float, float]:
        local_pos = local_trajectory[:, :2]  # only take the position info

        speed = center_state.dynamic_car_state.center_velocity_2d.magnitude()  # center forward speed

        controller = self.controllers[env_id][CBV_id]
        
        # get the PID control
        throttle, steer, brake = controller.control_pid(local_pos, speed)

        return throttle, steer, brake
    
    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: CarlaAgentState):

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        # force strict alignment
        delta = origin - global_trajectory[0, :2]
        adjusted_global_pos = global_trajectory[..., :2] + delta

        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(adjusted_global_pos - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)
