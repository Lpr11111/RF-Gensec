"""Frozen safety critic and RF-Gensec gate for PLUTO post-training."""
import torch
from rift.cbv.planning.rf_gensec.hjr import RiskHJR
from rift.cbv.planning.rf_gensec.risk_calculator import RiskCalculator
from rift.cbv.planning.rf_gensec.objectives import hard_gate_advantages, soft_gate_advantages


def load_safety(config):
    model = RiskHJR(state_dim=18, dims=[64, 64], action_dim=2,
                    barrier_type=config['hjr_barrier_type'],
                    barrier_delta=config.get('hjr_barrier_delta', 0.1),
                    risk_threshold=config.get('risk_threshold', 0.345))
    model.load_checkpoint(config['hjr_checkpoint'], load_q=False)
    return model.eval().requires_grad_(False)


def gated_advantage(reward_adv, value, next_value, risk, next_risk, undones,
                    hjr, config, gamma=0.98, lam=0.98):
    hazard, next_hazard = hjr.barrier_value(risk), hjr.barrier_value(next_risk)
    improving = ((next_risk <= hjr.risk_threshold) & (risk > hjr.risk_threshold)
                 if hjr.barrier_type == 'step' else next_hazard < hazard)
    q = torch.where(improving, torch.maximum(next_value, hazard), next_value)
    delta = value - q
    restoration = torch.empty_like(delta)
    carry = torch.zeros_like(delta[0])
    for i in reversed(range(len(delta))):
        carry = delta[i] + undones[i] * gamma * lam * carry
        restoration[i] = carry
    if config['gate_type'] == 'soft':
        return soft_gate_advantages(reward_adv, restoration, value, next_value,
                                    config['gate_temperature'], config['restoration_weight'])[0]
    if config['gate_type'] != 'hard':
        raise ValueError('gate_type must be hard or soft')
    return hard_gate_advantages(reward_adv, restoration, value, next_value,
                                config['restoration_weight'])[0]
