"""Train a risk-feasibility critic from externally supplied HDF5 transitions."""
import argparse
import json
from pathlib import Path
import sys

# Resolve the checkout before importing its package, including direct execution
# without an editable installation and execution from another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from rift.cbv.planning.rf_gensec.dataset import RFGensecDataset
from rift.cbv.planning.rf_gensec.hjr import RiskHJR
from rift.cbv.planning.rf_gensec.trainer import RiskHJRTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] /
                        'rift/cbv/planning/config/rf_gensec_hjr.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split', choices=['all', 'train', 'validation', 'test'])
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    steps = args.steps if args.steps is not None else cfg['train_steps']
    batch = args.batch_size if args.batch_size is not None else cfg['batch_size']
    if steps <= 0 or batch <= 0 or cfg['save_freq'] <= 0:
        parser.error('steps, batch size and save frequency must be positive')
    seed = int(cfg.get('seed', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = RFGensecDataset(args.data, actor_rows=cfg.get('actor_rows', 3),
                             split=args.split or cfg.get('dataset_split', 'train'),
                             split_seed=cfg.get('split_seed', seed),
                             split_fractions=cfg.get('split_fractions', [.7, .15, .15]),
                             risk_bin_edges=cfg.get('risk_bin_edges'),
                             risk_bin_fractions=cfg.get('risk_bin_fractions'))
    if dataset.length == 0:
        parser.error('selected dataset split is empty')
    state_dim = int(np.prod(dataset.obs_shape))
    expected = cfg.get('state_dim', state_dim)
    if state_dim != expected:
        parser.error(f'observation size {state_dim} differs from configured state_dim {expected}')
    model = RiskHJR(state_dim=state_dim, action_dim=dataset.action_dim,
                    **{key: cfg[key] for key in ['dims', 'gamma', 'tau', 'expectile',
                        'risk_threshold', 'penalty', 'barrier_type', 'barrier_delta'] if key in cfg})
    trainer = RiskHJRTrainer(model, dataset, args.output, lr=cfg.get('lr', 3e-4), device=args.device)
    if args.resume:
        trainer.load(args.resume)
    print(json.dumps(dataset.summary()))
    with (args.output / 'training.jsonl').open('a', encoding='utf-8') as log:
        for _ in range(steps):
            metrics = trainer.train_step(batch)
            log.write(json.dumps({'step': trainer.step, **metrics}) + '\n')
            if trainer.step % cfg['save_freq'] == 0:
                trainer.save()
    print(f'Saved {trainer.save()}')


if __name__ == '__main__':
    main()
