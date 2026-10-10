# RF-Gensec

Code for **Risk-Feasibility-Guided Safety-Critical Scenario Generation for
Autonomous Vehicle Testing**.

RF-Gensec combines a spatiotemporal vehicle risk field, an offline
Hamilton-Jacobi-reachability-inspired feasibility critic, and closed-loop
adaptation of trajectory-selection preferences. The pretrained PLUTO
trajectory-generation backbone remains frozen while the selection policy is
optimized with risk-feasibility guidance.

## Code structure

| Component | Directory |
| --- | --- |
| AV- and CBV-centered risk fields | `rift/cbv/planning/rf_gensec/risk_calculator.py` |
| Offline feasibility learning | `rift/cbv/planning/rf_gensec/` |
| Feasibility modulation and restoration | `rift/cbv/planning/fine_tuner/rlft/rlft_rf_gensec/safety.py` |
| Trajectory-selection fine-tuning | `rift/cbv/planning/fine_tuner/rlft/rlft_rf_gensec/` |
| Pretrained trajectory backbone and feature utilities | `rift/cbv/planning/pluto/` |
| Target-vehicle interface | `rift/ego/expert/` |

## Installation

For the risk-field and offline feasibility components:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Closed-loop integration uses the CARLA simulation environment and the PLUTO
trajectory model.

## Offline feasibility training

```bash
python scripts/train_hjr.py --data /path/to/transitions.hdf5 --output outputs/hjr --device cpu
```

Use `--device cuda` for GPU training and `--resume /path/to/model.torch` to
continue training. Run `python scripts/train_hjr.py --help` for the CLI options.

HDF5 transitions contain `obs`, `next_obs`, `actions`, and `dones`, with optional
`total_risk` or `risk` fields. Observations use vehicle features
`[x, y, length, width, relative_yaw, forward_speed]` in the CBV coordinate frame.
The route-goal row is excluded from vehicle risk calculations.

## Tests

```bash
python -m pytest -q
python scripts/check_release.py
```

The default test suite covers risk computation, feasibility learning,
modulation, and offline training. Simulator integration tests are located in
`tests/integration/`.

## Acknowledgments

This implementation builds on RIFT, PLUTO, and nuPlan-derived utilities.

