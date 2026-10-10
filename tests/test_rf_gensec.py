from pathlib import Path

import h5py
import numpy as np
import torch

from rift.cbv.planning.rf_gensec.dataset import RFGensecDataset
from rift.cbv.planning.rf_gensec.diagnostics import append_jsonl
from rift.cbv.planning.rf_gensec.hjr import RiskHJR
from rift.cbv.planning.rf_gensec.objectives import (
    adversarial_reward, hard_gate_advantages, normalise_advantage,
    soft_gate_advantages)
from rift.cbv.planning.rf_gensec.risk_calculator import RiskCalculator


def test_adversarial_reward_adds_av_risk_and_frea_environment_reward_equally():
    av_risk = torch.tensor([0.4, 0.2, 0.7])
    environment_reward = torch.tensor([0.5, -15.0, 15.0])

    reward = adversarial_reward(av_risk, environment_reward)

    assert torch.allclose(reward, torch.tensor([0.9, -14.8, 15.7]))


def test_center_cbv_risk_at_av_increases_when_av_is_closer():
    calculator = RiskCalculator()
    observation = np.zeros((2, 3, 6), dtype=np.float32)
    observation[:, 0] = [0.0, 0.0, 4.0, 2.0, 0.0, 8.0]
    observation[0, 1] = [20.0, 0.0, 4.0, 2.0, 0.0, 8.0]
    observation[1, 1] = [5.0, 0.0, 4.0, 2.0, 0.0, 8.0]
    values = calculator.calculate_center_risk_at_actor(observation)
    assert values.shape == (2,)
    assert 0.0 < values[0] < values[1] <= 1.0


def test_a0_step_barrier_at_threshold():
    """Eq. (7) is strict: R == R_thre is compliant, only R > R_thre violates."""
    model = RiskHJR(state_dim=18, action_dim=2, dims=(8, 8),
                    risk_threshold=0.345, penalty=4.0)
    risk = torch.tensor([0.3449, 0.3450, 0.3451])
    assert torch.equal(model.barrier_value(risk), torch.tensor([-1.0, -1.0, 4.0]))


def test_a0_bellman_target_contains_obstacle_maximum():
    model = RiskHJR(state_dim=18, action_dim=2, dims=(8, 8), gamma=0.98)
    hazard = torch.tensor([-1.0, 4.0, -1.0])
    next_value = torch.tensor([2.0, -3.0, 2.0])
    done = torch.tensor([0.0, 0.0, 1.0])
    target = model.bellman_target(hazard, next_value, done)
    expected = torch.tensor([1.94, 4.0, -1.0])
    assert torch.allclose(target, expected, atol=1e-6)


def test_a1_continuous_barrier_is_bounded_monotonic_and_zero_at_threshold():
    model = RiskHJR(state_dim=18, action_dim=2, dims=(8, 8),
                    risk_threshold=0.345, penalty=4.0,
                    barrier_type="continuous", barrier_delta=0.1)
    risk = torch.tensor([0.0, 0.245, 0.3449, 0.345, 0.3451, 0.445, 0.9])
    barrier = model.barrier_value(risk)
    assert barrier[0].item() == -1.0
    assert torch.allclose(barrier[3], torch.tensor(0.0), atol=1e-7)
    assert barrier[-1].item() == 4.0
    assert torch.all(barrier[1:] >= barrier[:-1])
    assert torch.allclose(barrier[2], torch.tensor(-0.001), atol=2e-7)
    assert torch.allclose(barrier[4], torch.tensor(0.001), atol=2e-7)


def test_barrier_configuration_validation():
    try:
        RiskHJR(barrier_type="unknown")
    except ValueError as error:
        assert "barrier_type" in str(error)
    else:
        raise AssertionError("unknown barrier type was accepted")
    try:
        RiskHJR(barrier_type="continuous", barrier_delta=0.0)
    except ValueError as error:
        assert "barrier_delta" in str(error)
    else:
        raise AssertionError("non-positive barrier delta was accepted")


def test_grouped_frea_hdf5_and_risk_alias(tmp_path):
    path = tmp_path / "frea.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(length=4, action_dim=3, obs_shape=(3, 6))
        group = handle.create_group("batch_0")
        group.attrs["length"] = 4
        group.create_dataset("obs", data=np.zeros((4, 3, 6), dtype=np.float32))
        group.create_dataset("next_obs", data=np.ones((4, 3, 6), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((4, 3), dtype=np.float32))
        group.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=np.float32))
        group.create_dataset("risk", data=np.linspace(0.1, 0.4, 4, dtype=np.float32))
    dataset = RFGensecDataset(path)
    assert dataset.length == 4
    assert dataset.obs_shape == (3, 6)
    assert dataset.action_dim == 3
    assert np.array_equal(dataset.data["total_risk"], dataset.data["risk"])
    batch = dataset.sample(3)
    assert batch["obs"].shape == (3, 3, 6)
    assert batch["actions"].shape == (3, 3)


def test_episode_split_has_no_transition_overlap(tmp_path):
    path = tmp_path / "episodes.hdf5"
    episode_count, episode_length = 20, 3
    length = episode_count * episode_length + 2
    identifiers = np.arange(length, dtype=np.float32)
    dones = np.zeros(length, dtype=np.float32)
    dones[episode_length - 1:episode_count * episode_length:episode_length] = 1.0
    with h5py.File(path, "w") as handle:
        group = handle.create_group("merged")
        group.create_dataset("obs", data=np.repeat(identifiers[:, None, None], 18, axis=1).reshape(length, 3, 6))
        group.create_dataset("next_obs", data=np.zeros((length, 3, 6), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((length, 2), dtype=np.float32))
        group.create_dataset("dones", data=dones)
        group.create_dataset("total_risk", data=np.zeros(length, dtype=np.float32))
    splits = {
        name: RFGensecDataset(path, split=name, split_seed=7)
        for name in ("train", "validation", "test")
    }
    ids = {name: set(dataset.data["obs"][:, 0, 0].astype(int)) for name, dataset in splits.items()}
    assert not (ids["train"] & ids["validation"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["validation"] & ids["test"])
    assert sum(len(value) for value in ids.values()) == episode_count * episode_length
    assert splits["train"].episode_count == 14
    assert splits["validation"].episode_count == 3
    assert splits["test"].episode_count == 3


def test_stratified_risk_sampling_respects_requested_counts(tmp_path):
    path = tmp_path / "risk_bins.hdf5"
    risk = np.array([0.1] * 10 + [0.25] * 10 + [0.4] * 10 + [0.7] * 10,
                    dtype=np.float32)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("merged")
        group.create_dataset("obs", data=np.zeros((40, 3, 6), dtype=np.float32))
        group.create_dataset("next_obs", data=np.zeros((40, 3, 6), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((40, 2), dtype=np.float32))
        group.create_dataset("dones", data=np.ones(40, dtype=np.float32))
        group.create_dataset("total_risk", data=risk)
    dataset = RFGensecDataset(
        path, risk_bin_edges=[-1, .2, .345, .5, 10],
        risk_bin_fractions=[.25, .30, .30, .15])
    batch = dataset.sample(20)
    values = batch["total_risk"].numpy()
    counts = [
        np.count_nonzero(values < .2),
        np.count_nonzero((values >= .2) & (values < .345)),
        np.count_nonzero((values >= .345) & (values < .5)),
        np.count_nonzero(values >= .5),
    ]
    assert counts == [5, 6, 6, 3]
    assert dataset.risk_bin_counts() == [10, 10, 10, 10]


def test_value_only_checkpoint_accepts_legacy_action_dimension(tmp_path):
    legacy = RiskHJR(state_dim=18, action_dim=3, dims=(8, 8))
    path = Path(tmp_path) / "legacy.torch"
    torch.save(legacy.checkpoint(step=10), path)
    online = RiskHJR(state_dim=18, action_dim=2, dims=(8, 8))
    payload = online.load_checkpoint(path, load_q=False)
    assert payload["step"] == 10
    for expected, actual in zip(legacy.v.parameters(), online.v.parameters()):
        assert torch.equal(expected, actual)


def test_checkpoint_rejects_barrier_semantics_mismatch(tmp_path):
    step_model = RiskHJR(state_dim=18, action_dim=2, dims=(8, 8), barrier_type="step")
    path = Path(tmp_path) / "step.torch"
    torch.save(step_model.checkpoint(step=10), path)
    continuous = RiskHJR(state_dim=18, action_dim=2, dims=(8, 8),
                         barrier_type="continuous", barrier_delta=0.1)
    try:
        continuous.load_checkpoint(path, load_q=False)
    except ValueError as error:
        assert "barrier_type" in str(error)
    else:
        raise AssertionError("incompatible barrier checkpoint was accepted")


def test_a0_hard_gate_is_finite_and_uses_realized_next_state():
    reward = torch.tensor([1.0, 2.0, 4.0, 8.0])
    restoration = torch.tensor([8.0, 4.0, 2.0, 1.0])
    value = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    next_value = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    result, safe, next_safe = hard_gate_advantages(reward, restoration, value, next_value)
    reward_norm = normalise_advantage(reward)
    restoration_norm = normalise_advantage(restoration)
    assert torch.allclose(result[0], reward_norm[0])
    assert torch.allclose(result[1:], restoration_norm[1:])
    assert safe.tolist() == [True, True, False, False]
    assert next_safe.tolist() == [True, False, True, False]
    assert torch.isfinite(result).all()


def test_soft_gate_is_continuous_and_balanced_at_boundary():
    reward = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    restoration = -reward
    value = torch.tensor([-0.2, -0.001, 0.0, 0.2])
    next_value = value.clone()
    result, gate = soft_gate_advantages(
        reward, restoration, value, next_value, temperature=0.1)
    assert torch.isfinite(result).all()
    assert torch.all(gate[:-1] > gate[1:])
    assert torch.allclose(gate[2], torch.tensor(0.5))
    assert 0.0 < gate[-1] < gate[0] < 1.0


def test_soft_gate_temperature_and_restoration_weight_validation():
    value = torch.arange(4, dtype=torch.float32)
    for kwargs, text in [({"temperature": 0.0}, "temperature"),
                         ({"restoration_weight": -1.0}, "restoration_weight")]:
        try:
            soft_gate_advantages(value, value, value, value, **kwargs)
        except ValueError as error:
            assert text in str(error)
        else:
            raise AssertionError(f"invalid {text} was accepted")


def test_adversarial_reward_source_is_cbv_and_receiver_is_av():
    """r_adv must be R_AV, not R_CBV (Eq. 16).

    The two objects are numerically distinguishable whenever the CBV and the AV
    move at different speeds: R_CBV attenuates with the *AV's* speed while R_AV
    attenuates with the *CBV's* speed. This pins the source/receiver direction
    so a future refactor cannot silently swap them back.
    """
    calculator = RiskCalculator()
    observation = np.zeros((1, 4, 6), dtype=np.float32)
    # CBV is fast, AV is slow, 10 m of longitudinal separation.
    observation[0, 0] = [0.0, 0.0, 4.5, 2.0, 0.0, 10.0]
    observation[0, 1] = [10.0, 0.0, 4.5, 2.0, 0.0, 5.0]
    observation[0, 3] = [20.0, 0.0, 2.0, 2.0, 0.0, 20.0]

    self_risk = calculator.calculate_dynamic_risk(observation, 3)
    av_risk = calculator.calculate_center_risk_at_actor(observation, actor_index=1)

    gap = 10.0 - 0.5 * 4.5
    expected_self = 1.0 / (gap / (0.631 * 5.0 + 1.0) + 1.0)   # AV speed decays
    expected_av = 1.0 / (gap / (0.631 * 10.0 + 1.0) + 1.0)    # CBV speed decays
    assert np.isclose(self_risk[0], expected_self, atol=1e-6)
    assert np.isclose(av_risk[0], expected_av, atol=1e-6)
    # The faster source spreads its field further, so R_AV must exceed R_CBV.
    assert av_risk[0] > self_risk[0]


def test_dynamic_risk_excludes_the_route_goal_row():
    """actor_rows must exclude row 3; it is a waypoint, not a vehicle."""
    calculator = RiskCalculator()
    observation = np.zeros((1, 4, 6), dtype=np.float32)
    observation[0, 0] = [0.0, 0.0, 4.5, 2.0, 0.0, 10.0]
    observation[0, 1] = [10.0, 0.0, 4.5, 2.0, 0.0, 5.0]
    observation[0, 3] = [4.0, 0.0, 2.0, 2.0, 0.0, 20.0]  # nearby goal waypoint
    vehicles_only = calculator.calculate_dynamic_risk(observation, 3)
    with_goal_row = calculator.calculate_dynamic_risk(observation, None)
    assert vehicles_only[0] < with_goal_row[0]


def test_lateral_decay_has_no_speed_term():
    """alpha_y is intentionally absent: the observation has no lateral speed."""
    calculator = RiskCalculator()
    assert not hasattr(calculator, "alpha_y")

    def lateral(speed):
        observation = np.zeros((1, 4, 6), dtype=np.float32)
        observation[0, 0] = [0.0, 0.0, 4.5, 2.0, 0.0, speed]
        observation[0, 1] = [0.0, 10.0, 4.5, 2.0, 0.0, 5.0]
        return calculator.calculate_center_risk_at_actor(observation, 1)[0]

    assert np.isclose(lateral(0.0), lateral(20.0), atol=1e-7)


def test_both_gates_share_a_common_advantage_scale():
    """Soft and hard gates must not differ in output scale.

    A convex blend of two standardised advantages has std sqrt(g^2+(1-g)^2) < 1,
    so without re-standardising, soft-gate runs would show smaller policy
    gradients than hard-gate runs for purely arithmetic reasons. That would
    confound the A0/A1 vs A2/A3 gradient-norm comparison with the gate effect.
    """
    generator = torch.Generator().manual_seed(0)
    reward = torch.randn(4096, generator=generator)
    restoration = torch.randn(4096, generator=generator)

    value = torch.full((4096,), -1.0)
    hard, _, _ = hard_gate_advantages(reward, restoration, value, value)
    assert abs(float(hard.std()) - 1.0) < 1e-2

    for temperature in (0.5, 1.0):
        for boundary in (-0.05, -1.0, -4.0):
            value = torch.full((4096,), boundary)
            soft, gate = soft_gate_advantages(
                reward, restoration, value, value, temperature)
            assert abs(float(soft.std()) - 1.0) < 1e-2, (
                f"soft gate scale drifted at tau={temperature}, V={boundary}: "
                f"std={float(soft.std()):.4f}, gate={float(gate.mean()):.4f}")


def test_hard_gate_applies_the_restoration_weight():
    """eta (Eq. 19) must act on the restoration branch of BOTH gate types."""
    reward = torch.tensor([1.0, 2.0, 4.0, 8.0])
    restoration = torch.tensor([8.0, 4.0, 2.0, 1.0])
    value = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    next_value = torch.tensor([-1.0, 1.0, -1.0, 1.0])

    baseline, _, _ = hard_gate_advantages(reward, restoration, value, next_value,
                                         restoration_weight=1.0)
    scaled, _, _ = hard_gate_advantages(reward, restoration, value, next_value,
                                        restoration_weight=0.5)
    # Adversarial branch is untouched; restoration branch is halved.
    assert torch.allclose(scaled[0], baseline[0])
    assert torch.allclose(scaled[1:], 0.5 * baseline[1:])

    try:
        hard_gate_advantages(reward, restoration, value, next_value,
                             restoration_weight=-1.0)
    except ValueError as error:
        assert "restoration_weight" in str(error)
    else:
        raise AssertionError("negative restoration_weight was accepted")


def test_local_diagnostics_are_append_only_valid_json(tmp_path):
    import json

    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"update_step": 1, "actor_grad_norm": 0.5})
    append_jsonl(path, {"update_step": 2, "actor_grad_norm": 0.25})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["update_step"] for record in records] == [1, 2]
    assert records[-1]["actor_grad_norm"] == 0.25
