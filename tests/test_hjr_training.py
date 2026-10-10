import h5py
import numpy as np
import torch
import subprocess
import sys
from pathlib import Path

from rift.cbv.planning.rf_gensec.dataset import RFGensecDataset
from rift.cbv.planning.rf_gensec.hjr import RiskHJR
from rift.cbv.planning.rf_gensec.trainer import RiskHJRTrainer


def test_training_updates_critic_and_resume_restores_optimizer(tmp_path):
    path = tmp_path / 'transitions.hdf5'
    rng = np.random.default_rng(7)
    with h5py.File(path, 'w') as file:
        file['obs'] = rng.normal(size=(16, 3, 6)).astype(np.float32)
        file['next_obs'] = rng.normal(size=(16, 3, 6)).astype(np.float32)
        file['actions'] = rng.normal(size=(16, 2)).astype(np.float32)
        file['dones'] = np.ones(16, dtype=np.float32)
        file['total_risk'] = np.linspace(0, .8, 16, dtype=np.float32)
    dataset = RFGensecDataset(path)
    model = RiskHJR(dims=[8, 8])
    original = [value.detach().clone() for value in model.parameters()]
    trainer = RiskHJRTrainer(model, dataset, tmp_path / 'output')
    metrics = trainer.train_step(8)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(original, model.parameters()))
    saved = trainer.save()
    resumed = RiskHJRTrainer(RiskHJR(dims=[8, 8]), dataset, tmp_path / 'resume')
    resumed.load(saved)
    assert resumed.step == 1
    for name in ['q', 'v']:
        for expected, actual in zip(getattr(model, name).parameters(), getattr(resumed.model, name).parameters()):
            assert torch.equal(expected, actual)
    assert resumed.q_optimizer.state
    assert resumed.v_optimizer.state
    resumed.train_step(8)
    assert resumed.step == 2


def test_cli_trains_from_another_working_directory(tmp_path):
    path = tmp_path / 'input.hdf5'
    with h5py.File(path, 'w') as file:
        file['obs'] = np.zeros((8, 3, 6), dtype=np.float32)
        file['next_obs'] = np.zeros((8, 3, 6), dtype=np.float32)
        file['actions'] = np.zeros((8, 2), dtype=np.float32)
        file['dones'] = np.ones(8, dtype=np.float32)
        file['total_risk'] = np.linspace(0, .8, 8, dtype=np.float32)
    config = tmp_path / 'config.yaml'
    config.write_text('train_steps: 2\nbatch_size: 4\nsave_freq: 2\nstate_dim: 18\ndims: [8, 8]\ndataset_split: all\n')
    script = Path(__file__).resolve().parents[1] / 'scripts/train_hjr.py'
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, str(script), '--data', str(path),
                             '--config', str(config), '--output', str(output)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / 'model.HJR.0000002.torch').is_file()
    assert len((output / 'training.jsonl').read_text().splitlines()) == 2
