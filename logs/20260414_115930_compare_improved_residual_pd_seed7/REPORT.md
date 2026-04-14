# SAC Tracking Debug Report

## Current Pipeline

- `TwoLink` in `Core_SAC.py` owns dynamics, inverse kinematics, forward kinematics, PD torque, and pygame drawing.
- The previous control loop mixed target generation, SAC updates, reward calculation, checkpointing, and pygame rendering in one 60 Hz loop.
- Circle targets are defined by radius, center offset, and angular rate; IK converts target end-effector points to joint targets.
- `sac_agent.py` contains the generic SAC actor, critics, replay buffer, updates, and checkpoint serialization.

## Likely Failure Points Found

- The code described `SAC(delta_q) + PD`, but SAC output was applied directly as torque and `pd_torque()` was unused.
- Training was coupled to pygame rendering and `clock.tick(60)`, slowing data collection.
- Observations mixed radians, rad/s, and raw pixel coordinates without normalization.
- The reward was dominated by pixel-scale tracking error and no metrics were persisted.
- There was no deterministic evaluation entry point, best checkpoint, config snapshot, or trajectory plot.

## Changes

- Added `train`, `eval`, `render`, and `compare` subcommands.
- Added timestamped `logs/` run folders with config snapshots, JSONL/CSV logs, checkpoints, summaries, and plots.
- Added deterministic evaluation with per-episode trajectory CSV files and target-vs-actual plots.
- Added normalized observations and meter-scale reward for the improved variant.
- Added `residual_pd` mode so SAC learns bounded residual torque on top of the existing PD baseline.

## Controlled Comparison

- Legacy direct-torque untrained mean error: `0.200077 m`
- Improved residual-PD untrained mean error: `0.050482 m`
- Mean-error ratio legacy/improved: `3.96x`
- Legacy success rate: `0.013`
- Improved success rate: `0.198`

## Remaining Work

- Longer training is still needed before claiming convergence of the residual SAC policy.
- Next tuning should focus on residual action limits, reward weights, and target-velocity tracking.
