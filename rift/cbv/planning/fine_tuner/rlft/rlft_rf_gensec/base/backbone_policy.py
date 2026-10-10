# Adapted for RF-Gensec: retain method infrastructure and remove comparison-policy behavior.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : rlft_pluto.py
@Date    : 2025/1/24
'''
import copy
import json
import gc
import re
import wandb
import hydra
import numpy as np
import torch
import hydra._internal.instantiate._instantiate2
import hydra.types

from pathlib import Path
from typing import Any, Dict, List
from rift.cbv.planning.fine_tuner.training_builder import TrainingEngine, build_training_engine
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.cbv.planning.pluto.model.pluto_model import PlanningModel
from rift.cbv.planning.pluto.backbone import TrajectoryBackbone
from rift.cbv.planning.route_planner.route_planner import CBVRoutePlanner
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.util.torch_util import CUDA

# Instantiation related symbols
instantiate = hydra._internal.instantiate._instantiate2.instantiate


class BackbonePolicy(TrajectoryBackbone):
    name = 'rf_gensec_backbone_policy'
    type = 'rlft'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.config_path = config['rlft_config_path']
        self.config_name = config['rlft_config_name']
        self.model_path = Path(config['ROOT_DIR']) / config['model_path']
        self.cbv_recog = config['cbv_recog']
        self.seed = config['seed']
        self.pretrain_seed = config['pretrain_seed']
        self.checkpoint = None

        hydra.initialize(config_path=self.config_path, version_base=None)
        self.cfg = hydra.compose(
            config_name=self.config_name,
            overrides=[
                f"+output_dir={json.dumps(str(self.logger.output_dir), ensure_ascii=False)}",
                f"frame_rate={self._frame_rate}",
                f"+name={json.dumps(self.logger.output_dir.name, ensure_ascii=False)}",
            ]
        )
        self.initial_lr = self.cfg.lr  # initial learning rate

        if config['mode'] == 'train_cbv':
            self.load_agent_info = config['ego_policy'] + '-' + str(self.cbv_recog) + '-seed' + str(self.seed)
        else:
            self.load_agent_info = config['pretrain_ego'] + '-' + str(self.cbv_recog) + '-seed' + str(self.pretrain_seed)

        self.save_agent_info = config['ego_policy'] + '-' + str(self.cbv_recog) + '-seed' + str(self.seed)

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
            # build the train model
            self.train_model = CUDA(PlanningModel(radius=self.radius))
            self.train_model.train()
            # inference model
            self.pluto_model.eval()
        elif mode == 'eval':
            self.pluto_model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, List[Dict[Any, Any]]]:
        CBVs_action = [{} for _ in range(self.num_scenario)]
        CBVs_old_log_prob = [{} for _ in range(self.num_scenario)]
        CBVs_action_mode = [{} for _ in range(self.num_scenario)]

        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            if not CBVs_obs:
                continue
            
            env_id = info['env_id']
            valid_items = []
            for CBV_id, CBV_obs in CBVs_obs.items():
                feature = CBV_obs['raw_pluto_feature']
                error = feature.validation_error()
                if error is None:
                    valid_items.append((CBV_id, CBV_obs))
                    continue

                # Keep CARLA controllable for this last frame, but deliberately
                # omit PPO metadata.  CBVRolloutBuffer will discard the invalid
                # trajectory instead of training on a fabricated action.
                CBVs_action[env_id][CBV_id] = (0.0, 0.0, 1.0)
                self.logger.log(
                    f'>> Skipping invalid PLUTO feature for CBV:{CBV_id} '
                    f'(env={env_id}): {error}; applying emergency brake',
                    color='yellow',
                )

            if not valid_items:
                continue

            pluto_feature_data = PlutoFeature.collate([
                CBV_obs['raw_pluto_feature'] for _, CBV_obs in valid_items
            ]).to_device(self.device).data
            with torch.no_grad():
                pluto_output = self.pluto_model(pluto_feature_data)
            for index, (CBV_id, CBV_obs) in enumerate(valid_items):
                # get the action for the inference
                (
                    CBVs_action[env_id][CBV_id],
                    CBVs_old_log_prob[env_id][CBV_id],
                    CBVs_action_mode[env_id][CBV_id]
                ) = self._get_action(pluto_feature_data, pluto_output, CBV_obs, env_id, CBV_id, index)

        # Keep the frozen RF/HJR evaluator policy-independent.  This is a
        # no-op outside evaluation and for policies that own their diagnostics.
        self._update_frozen_eval_diag(CBVs_obs_list, infos)
            
        # render part
        if self._render:
            for info in infos:
                env_id = info['env_id']
                ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
                ego_state = CarlaDataProvider.get_current_state(ego)
                nearby_agents = CarlaDataProvider.get_ego_nearby_agents(ego.id)
                nearby_agents_states = {agent.id: CarlaDataProvider.get_current_state(agent) for agent in nearby_agents}

                self._render_data[env_id].update({
                    'ego_states': {ego.id: ego_state},
                    'nearby_agents_states': nearby_agents_states
                })
        

        # clean useless CBVs
        self._clean_CBVs(infos, CBVs_obs_list)
        
        data = {
            'CBVs_actions': CBVs_action,
            'CBVs_actions_old_log_prob': CBVs_old_log_prob,
            'CBVs_actions_mode': CBVs_action_mode
        }

        return data


    def set_route_planner(self, route_planner:CBVRoutePlanner):
        self.route_planner = route_planner

    def get_render_data(self, env_id):
        return self._render_data[env_id]

    def set_buffer(self, buffer, total_routes):
        self.buffer: CBVRolloutBuffer = buffer


    def update_training_ckpt(self):
        load_dir = self.model_path / self.load_agent_info
        try:
            pattern = re.compile(r"carla_episode=(\d+)")
            episode_files = sorted(
                load_dir.glob("*.ckpt"),
                key=lambda file: int(pattern.search(file.stem).group(1)),
                reverse=True
            )
            self.current_epoch = len(episode_files)
            if episode_files:
                self.checkpoint = episode_files[0].as_posix()
                self.logger.log(f"Updated checkpoint: {episode_files[0].name}", 'green')
            else:
                self.logger.log("No checkpoint files found.", 'yellow')

        except (AttributeError, ValueError, Exception) as e:
            self.logger.log(f"Error processing checkpoint files: {e}", 'red')
            raise

    def load_model(self, resume=True):
        load_dir = self.model_path / self.load_agent_info
        episode_files = list(load_dir.glob(f"*.ckpt"))

        if resume and episode_files:
            latest_file = max(
                episode_files,
                key=lambda file: int(re.search(r"carla_episode=(\d+)", file.stem).group(1))
            )
            self.checkpoint = latest_file.as_posix()  # use the latest model
            self.continue_episode = int(re.search(r"carla_episode=(\d+)", latest_file.stem).group(1))
            self.current_epoch = len(episode_files)  # update the current epoch
            self.logger.log(f">> Loading {self.name} model from {latest_file.name}", 'yellow')
        else:
            if not resume:
                self.logger.log(f">> Clearning all the data in {load_dir}", 'red')
                for file in episode_files:
                    file.unlink()  # remove all the model files
            self.checkpoint = self.config['ckpt_path']  # use the pretrained checkpoint of pluto
            self.logger.log(f">> Using Pluto pretrained model", 'yellow')
            self.continue_episode = 0
            self.current_epoch = 0
        
        # the inference model
        self.pluto_model.load_state_dict(self.load_infer_checkpoint(self.checkpoint, device_name=self.device))

    def save_model(self, episode):
        pass

    def finish(self):
        if wandb.run is not None:
            wandb.finish()
