from types import SimpleNamespace
from pathlib import Path
import pytest
import numpy as np
import torch
from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.observation import penalized_collision
from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.safety import gated_advantage
from rift.cbv.planning.rf_gensec.hjr import RiskHJR
from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.policy import RLFTRFGensec


def test_collision_excludes_ego_and_nonvehicles():
    assert not penalized_collision({'other_actor_id': 1}, 1, 'vehicle.car')
    assert penalized_collision({'other_actor_id': 2}, 1, 'vehicle.car')
    assert not penalized_collision({'other_actor_id': 2}, 1, 'walker.pedestrian')
    assert not penalized_collision({'other_actor_id': 2}, 1, 'static.wall')
    assert not penalized_collision(None, 1, '')


def test_dense_collision_reward_excludes_av_even_if_actor_was_removed(monkeypatch):
    from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec import observation as module
    from rift.gym_carla.reward.reward_model import DenseRewardModel
    reward = module.RIFTEnvironmentReward.__new__(module.RIFTEnvironmentReward)
    reward.ego_vehicle = SimpleNamespace(id=1)
    reward.angular_history = {}
    reward.map = SimpleNamespace(get_waypoint=lambda *a, **kw: object())
    reward.reward_model = DenseRewardModel()
    reward.get_delta_info = lambda _: (0., 0.)
    actor = SimpleNamespace(get_angular_velocity=lambda: SimpleNamespace(z=0.))
    monkeypatch.setattr(module.CDP, 'get_CBVs_by_ego', lambda _: {7: actor})
    monkeypatch.setattr(module.CDP, 'get_world', lambda: SimpleNamespace(get_actor=lambda _: None))
    monkeypatch.setattr(module.CDP, 'get_location', lambda _: object())
    monkeypatch.setattr(module.CDP, 'get_velocity', lambda _: SimpleNamespace(length=lambda: 5.))
    monkeypatch.setattr(module.CDP, 'get_acceleration', lambda _: SimpleNamespace(length=lambda: 0.))
    monkeypatch.setattr(module.GameTime, 'get_time', lambda: 1.)
    baseline = reward.get_reward({7: None})[7]
    ego_collision = reward.get_reward({7: {'other_actor_id': 1, 'other_actor_type': 'vehicle.car'}})[7]
    other_collision = reward.get_reward({7: {'other_actor_id': 2, 'other_actor_type': 'vehicle.car'}})[7]
    assert ego_collision == baseline
    assert np.isclose(other_collision - baseline, -25.)


def test_gate_has_finite_normalized_advantages_and_respects_episode_boundaries():
    hjr = RiskHJR(dims=[8, 8])
    cfg = dict(gate_type='soft', gate_temperature=0.5, restoration_weight=1.)
    x = torch.tensor([0.1, 0.6, -0.3, 0.7])
    adv = gated_advantage(x, x, x.flip(0), x.abs(), x.abs().flip(0),
                          torch.zeros(4), hjr, cfg)
    assert torch.isfinite(adv).all()
    assert abs(adv.mean()) < 1e-5
    assert abs(adv.std() - 1) < 1e-3


def test_a3_configuration_uses_continuous_soft_gate():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / 'rift/cbv/planning/config/rlft_rf_gensec_a3.yaml')
    assert cfg.policy_name == 'rlft_rf_gensec'
    assert cfg.hjr_barrier_type == 'continuous'
    assert cfg.gate_type == 'soft'
    assert cfg.gate_temperature == 1.0
    assert cfg.hjr_barrier_delta == 0.1
    assert cfg.model_path.endswith('rlft_rf_gensec_a3')


def test_a0_configuration_uses_step_hard_gate():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / 'rift/cbv/planning/config/rlft_rf_gensec_a0.yaml')
    assert cfg.policy_name == 'rlft_rf_gensec'
    assert cfg.hjr_barrier_type == 'step'
    assert cfg.gate_type == 'hard'
    assert cfg.restoration_weight == 1.0
    assert cfg.model_path.endswith('rlft_rf_gensec_a0')
    assert cfg.buffer_capacity == 512
    assert cfg.batch_size == 16
    assert cfg.wandb_mode == 'disabled'


def test_action_log_probability_matches_full_masked_training_distribution(monkeypatch):
    from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec import policy as module
    policy = RLFTRFGensec.__new__(RLFTRFGensec)
    policy.mode, policy._render = 'eval', False
    policy.get_control = lambda *args: (0.2, 0., 0.)
    monkeypatch.setattr(module.CDP, 'get_actor_by_id', lambda _: object())
    monkeypatch.setattr(module.CDP, 'get_history_state', lambda _: [object()])
    logits = torch.tensor([[[100., 100.], [1., 3.], [0., 2.]]])
    features = {'reference_line': {'valid_mask': torch.tensor([[[False], [True], [True]]])}}
    output = {'probability': logits, 'candidate_trajectories': torch.zeros(1, 3, 2, 8, 3)}
    _, old_log_prob, mode = policy._get_action(features, output, {}, 0, 2, 0)
    assert mode == (1, 1)  # Retains the original padded reference-line index.
    expected = torch.log_softmax(logits[0].masked_fill(
        torch.tensor([[True], [False], [False]]), -1e8).flatten(), 0)[3]
    assert np.isclose(old_log_prob, expected.item())
    assert np.isclose(np.exp(expected.item() - old_log_prob), 1.)


def test_real_checkpoint_loading_and_trainable_layers(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from hydra.core.global_hydra import GlobalHydra
    from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.trainer import LightningTrainer
    root = Path(__file__).resolve().parents[2]
    hjr = root / 'experiments/rf_gensec/formal_checkpoints/hjr/step/seed0/model.HJR.2000000.torch'
    pluto = root / 'rift/cbv/planning/model_ckpt/pluto/pluto_1M_aux_cil.ckpt'
    if not hjr.exists() or not pluto.exists():
        pytest.skip('Local pretrained checkpoints not installed')
    monkeypatch.setenv('MODEL_DEVICE', 'cpu')
    cfg = OmegaConf.to_container(OmegaConf.load(root / 'rift/cbv/planning/config/rlft_rf_gensec.yaml'))
    cfg.update(ROOT_DIR=str(root), hjr_checkpoint=str(hjr), ckpt_path=str(pluto),
               model_path=str(tmp_path / 'models'), mode='train_cbv', num_scenario=1,
               frame_rate=10, need_video_render=False, cbv_recog='rule', seed=0,
               pretrain_seed=0, ego_policy='expert', pretrain_ego='expert')
    logger = SimpleNamespace(output_dir=tmp_path / 'RIFT（risk）', log=lambda *a, **kw: None)
    GlobalHydra.instance().clear()
    try:
        policy = RLFTRFGensec(cfg, logger)
        policy.load_model()
        policy.set_mode('train')
        trainer = LightningTrainer(policy.train_model, lr=1e-4, cl_lr_decay=.9,
            weight_decay=1e-5, epochs=16, warmup_epochs=3, frame_rate=10,
            trainable_layers=list(policy.cfg.trainable_layers), use_drivable_area_loss=False)
        missing, unexpected = trainer.load_state_dict(
            policy.load_train_checkpoint(policy.checkpoint, device_name='cpu'), strict=False)
        assert not unexpected
        assert missing and all(k.startswith('model.value_net.') for k in missing)
        trainable = [k for k, p in policy.train_model.named_parameters() if p.requires_grad]
        assert trainable
        assert all(k.startswith(('planning_decoder.pi_head.', 'value_net.')) for k in trainable)
        assert any(k.startswith('value_net.') for k in trainable)
        assert all(not p.requires_grad for p in policy.hjr.parameters())
        assert policy.cfg.datamodule.type == 'rlft-rf-gensec'
        # Exercise the actual PPO objective and backward on both trainable heads.
        logits = policy.train_model.planning_decoder.pi_head(torch.randn(4, 2, 3, 128)).squeeze(-1)
        modes = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 2]])
        old = torch.log_softmax(logits.detach().flatten(1), 1)[torch.arange(4), modes[:, 0] * 3 + modes[:, 1]]
        batch = dict(state_torch=torch.randn(4, 128), advantage_torch=torch.tensor([-1., .2, .3, .5]),
                     reward_sum_torch=torch.ones(4), old_log_prob_torch=old)
        loss = trainer.get_ppo_loss(logits, modes, batch)
        assert torch.isfinite(loss)
        loss.backward()
        for prefix in ('planning_decoder.pi_head.', 'value_net.'):
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for k, p in policy.train_model.named_parameters() if k.startswith(prefix))
        assert all(p.grad is None for p in policy.train_model.parameters() if not p.requires_grad)
    finally:
        GlobalHydra.instance().clear()
