"""PLUTO trajectory-selection PPO with RF-Gensec feasibility advantages."""
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.base.policy import TrajectorySelectionPolicy
from rift.scenario.tools.carla_data_provider import CarlaDataProvider as CDP
from .safety import load_safety, RiskCalculator


class RLFTRFGensec(TrajectorySelectionPolicy):
    name = 'rlft_rf_gensec'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        if not config.get('hjr_checkpoint'):
            raise ValueError('rlft_rf_gensec requires --hjr-checkpoint')
        safety = {k: config[k] for k in ('gate_type', 'gate_temperature', 'restoration_weight',
                  'hjr_barrier_type', 'hjr_barrier_delta', 'risk_threshold', 'risk_field')}
        safety['hjr_checkpoint'] = str(Path(config['ROOT_DIR']) / config['hjr_checkpoint'])
        self.hjr = load_safety(safety)
        self.risk = RiskCalculator(**config['risk_field'])
        OmegaConf.update(self.cfg, 'datamodule.type', 'rlft-rf-gensec')
        OmegaConf.update(self.cfg, 'datamodule.rf_gensec', safety, force_add=True)
        OmegaConf.update(self.cfg, 'custom_trainer._target_',
                        'rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.trainer.LightningTrainer')
        # Smaller batches bound PLUTO feature memory; train only the pi head and critic.
        self.cfg.datamodule.train_batch_size = config.get('batch_size', 16)
        self.cfg.datamodule.val_batch_size = config.get('batch_size', 16)
        if config.get('wandb_mode'):
            OmegaConf.update(self.cfg, 'wandb.mode', config['wandb_mode'])
        self.diag = {}

    def load_model(self, resume=True):
        directory = self.model_path / self.load_agent_info
        if self.config['mode'] == 'eval' and not list(directory.glob('*.ckpt')):
            raise FileNotFoundError(f'No fine-tuned rlft_rf_gensec checkpoint in {directory}')
        if not resume and list(directory.glob('*.ckpt')):
            raise ValueError('Use a new model directory for a fresh training run')
        super().load_model(resume)

    def get_action(self, observations, infos, deterministic=False):
        self.diag = {}
        self._deterministic = deterministic
        data = super().get_action(observations, infos, deterministic)
        for obs, info in zip(observations, infos):
            if not obs:
                continue
            states = np.stack([o['rf_state'] for o in obs.values()])
            risk = self.risk.calculate_dynamic_risk(states, 3)
            av = self.risk.calculate_center_risk_at_actor(states, 1)
            with torch.no_grad():
                values = self.hjr.feasibility_value(torch.from_numpy(states)).numpy()
            threshold = self.hjr.risk_threshold
            self.diag[info['env_id']] = {'risk_threshold': self.hjr.risk_threshold,
                'self_risk_sum': float(risk.sum()), 'self_risk_count': len(risk),
                'self_risk_max': float(risk.max()),
                'self_risk_violation_count': int((risk > self.hjr.risk_threshold).sum()),
                'self_risk_overshoot_max': float(np.maximum(risk - threshold, 0).max()),
                'target_quadrant_count': int(((av > threshold) & (risk <= threshold)).sum()),
                'suicide_quadrant_count': int(((av > threshold) & (risk > threshold)).sum()),
                'safe_quadrant_count': int(((av <= threshold) & (risk <= threshold)).sum()),
                'self_only_risk_quadrant_count': int(((av <= threshold) & (risk > threshold)).sum()),
                'av_risk_sum': float(av.sum()), 'av_risk_max': float(av.max()), 'av_risk_count': len(av),
                'hjr_value_sum': float(values.sum()), 'hjr_value_max': float(values.max()),
                'hjr_value_count': len(values), 'hjr_infeasible_count': int((values > 0).sum()),
                'per_actor': {str(k): {'cbv_risk': float(risk[i]), 'av_risk': float(av[i]),
                                     'hjr_value': float(values[i])} for i, k in enumerate(obs)}}
        return data

    def get_eval_diag(self, env_id):
        return {'rf_gensec': self.diag.get(env_id, {})}

    def _get_action(self, features, output, obs, env_id, actor_id, index):
        logits = output['probability'][index]
        valid = features['reference_line']['valid_mask'][index].any(-1)
        if not valid.any():
            raise RuntimeError('PLUTO has no valid reference-line candidates')
        logits = logits.masked_fill(~valid[:, None], -1e8)
        distribution = torch.distributions.Categorical(logits=logits.flatten())
        selected = (distribution.sample() if self.mode == 'train' and not
                    getattr(self, '_deterministic', False) else logits.flatten().argmax())
        r, m = divmod(int(selected), logits.shape[1])
        log_prob = float(distribution.log_prob(selected))
        trajectory = output['candidate_trajectories'][index, r, m].cpu().numpy().astype(np.float64)
        state = CDP.get_history_state(CDP.get_actor_by_id(actor_id))[-1]
        control = self.get_control(env_id, actor_id, state, trajectory)
        if self._render:
            angle = state.rear_axle.heading
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            global_traj = trajectory.copy()
            global_traj[:, :2] = trajectory[:, :2] @ rotation + state.rear_axle.array
            global_traj[:, 2] += angle
            render = self._render_data[env_id]
            render['CBV_states'][actor_id] = state
            for key in ('route_ids', 'reference_lines', 'route_waypoints', 'interaction_wp'):
                render[key + '_list'].append(obs[key])
            render['planning_trajectory_list'].append(global_traj)
            render['candidate_trajectories_list'].append(global_traj[None])
            render['candidate_index_list'].append(0)
            render['predictions_list'].append(None)
        return control, log_prob, (r, m)
