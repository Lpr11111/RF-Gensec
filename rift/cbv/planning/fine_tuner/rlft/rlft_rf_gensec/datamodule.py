"""PPO returns from R_AV + RIFT reward, actor advantages from the HJR gate."""
import numpy as np
import torch
from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.base.datamodule import TrajectorySelectionDataModule
from .safety import load_safety, gated_advantage, RiskCalculator


class RFGensecDataModule(TrajectorySelectionDataModule):
    def preprocess_buffer(self, model):
        safety = self.cfg.rf_gensec
        hjr = load_safety(safety)
        calculator = RiskCalculator(**dict(safety.risk_field))
        obs = np.stack([o['rf_state'] for o in self.buffer.get_key_data('CBVs_obs')])
        nxt = np.stack([o['rf_state'] for o in self.buffer.get_key_data('CBVs_next_obs')])
        av_risk = calculator.calculate_center_risk_at_actor(obs, actor_index=1)
        original = self.buffer.buffer_data['CBVs_reward']
        self.buffer.buffer_data['CBVs_reward'] = (
            np.asarray(original, dtype=np.float32).reshape(-1) + av_risk)
        # Frozen encoder and reference probability distribution must use eval
        # mode, so dropout does not perturb the PPO probability ratio.
        model.eval()
        try:
            super().preprocess_buffer(model)
        finally:
            self.buffer.buffer_data['CBVs_reward'] = original
        with torch.no_grad():
            device = next(model.parameters()).device
            states = self.buffer.get_key_data('CBVs_state').to(device)
            reward_adv = self.buffer.get_key_data('CBVs_reward_sum') - model.value_net(states).cpu()
            value = hjr.feasibility_value(torch.from_numpy(obs))
            next_value = hjr.feasibility_value(torch.from_numpy(nxt))
            risk = torch.as_tensor(calculator.calculate_dynamic_risk(obs, 3))
            next_risk = torch.as_tensor(calculator.calculate_dynamic_risk(nxt, 3))
            undones = 1 - torch.as_tensor(np.asarray(self.buffer.get_key_data('CBVs_done')), dtype=torch.float32).reshape(-1)
            advantages = gated_advantage(reward_adv, value, next_value, risk, next_risk,
                                         undones, hjr, safety, self.gamma, self.lambda_gae_adv)
        self.buffer.add_extra_data({'CBVs_advantage': advantages.cpu()})
        print(f'>> rlft_rf_gensec: R_AV={av_risk.mean():.6f}, '
              f'R_RIFT={np.asarray(original).mean():.6f}, '
              f'R_total={(np.asarray(original).reshape(-1) + av_risk).mean():.6f}, '
              f'HJR_infeasible={(value > 0).float().mean().item():.6f}', flush=True)
