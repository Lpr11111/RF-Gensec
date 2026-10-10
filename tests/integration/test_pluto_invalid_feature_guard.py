import numpy as np
import torch

from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer


def _feature(reference_valid=True, finite=True):
    value = 0.0 if finite else np.nan
    return PlutoFeature(data={
        'current_state': torch.tensor([value], dtype=torch.float32),
        'agent': {'valid_mask': torch.tensor([[True]])},
        'reference_line': {'valid_mask': torch.tensor([[reference_valid]])},
    })


def test_pluto_feature_rejects_all_masked_and_nonfinite_input():
    assert _feature().is_valid
    assert _feature(reference_valid=False).validation_error() == \
        'no valid reference-line point'
    assert _feature(finite=False).validation_error() == \
        'non-finite tensor at data.current_state'


def test_invalid_terminal_feature_discards_partial_on_policy_trajectory():
    keys = [
        'CBVs_actions', 'CBVs_actions_old_log_prob', 'CBVs_actions_mode',
        'CBVs_obs', 'CBVs_next_obs', 'CBVs_reward', 'CBVs_terminated',
        'CBVs_done',
    ]
    buffer = CBVRolloutBuffer(
        num_scenario=1,
        mode='train_cbv',
        cbv_config={'buffer_capacity': 8, 'data_keys': keys},
    )
    for values in buffer.temp_buffer.values():
        values[7].append('previous-frame')

    valid_obs = {'raw_pluto_feature': _feature()}
    invalid_obs = {'raw_pluto_feature': _feature(reference_valid=False)}
    data = {
        'CBV_ids': [[7]],
        'CBVs_actions': [{7: (0.0, 0.0, 1.0)}],
        'CBVs_actions_old_log_prob': [{7: 0.0}],
        'CBVs_actions_mode': [{7: (0, 0)}],
        'CBVs_obs': [{7: valid_obs}],
        'CBVs_next_obs': [{7: invalid_obs}],
        'CBVs_reward': [{7: -1.0}],
        'CBVs_terminated': [{7: True}],
        'CBVs_done': [{7: True}],
    }
    processed, length = buffer.process_data_dict(data)

    assert length == 0
    assert all(7 not in values for values in buffer.temp_buffer.values())
    assert all(not values for values in processed.values())
