# SAC Trajectory Tracking Report

## Current Pipeline

- `Core_SAC.py` contains the two-link arm dynamics, inverse kinematics, forward kinematics, target generation, reward/metric calculation, pygame rendering, and the new train/eval/compare entry points.
- `sac_agent.py` contains the generic SAC actor, twin critics, replay buffer, update logic, entropy tuning, and checkpoint serialization.
- The target trajectory is a circle generated from `radius_px`, `center_offset_px`, and `omega`. The target end-effector point is converted to desired joint position and velocity through IK.
- The improved controller uses the existing PD torque as a stabilizing baseline and trains SAC to output bounded residual torque.

## Main Issues Found

- The original pygame loop title/comment said `SAC(delta_q) + PD`, but the SAC action was applied directly as torque and `pd_torque()` was unused.
- Training, evaluation, target generation, checkpointing, and rendering were coupled in one 60 Hz pygame loop.
- The legacy observation mixed radians, rad/s, and raw pixel coordinates without normalization.
- The legacy reward used pixel-scale tracking error, which dominated the reward and made metric interpretation harder.
- There was no deterministic evaluation path, config snapshot, best checkpoint, trajectory plot, or persistent tracking metric log.
- The actor mean head used default random initialization, so deterministic residual control started with arbitrary residual torque instead of the PD baseline.

## Changes Made

- Added CLI subcommands:
  - `train`
  - `eval`
  - `render`
  - `compare`
- Added timestamped `logs/` run directories with:
  - `config.json`
  - `training_episodes.csv`
  - `training_steps.jsonl`
  - `evaluation.csv`
  - per-evaluation trajectory CSV files
  - trajectory plots
  - training/evaluation plots
  - `summary.md`
  - checkpoints including `best.pt` and `final.pt`
- Added `legacy` and `improved` variants for ablation.
- Added `direct_torque` and `residual_pd` control modes.
- Added normalized improved observations using joint trigonometric features, normalized velocities, joint tracking error, and end-effector tracking error.
- Added meter-scale improved reward with tracking, progress, joint-error, effort, and success terms.
- Initialized the SAC actor output head near zero so residual-PD evaluation starts from the PD baseline while SAC still explores stochastically.
- Fixed `.gitignore` to ignore `__pycache__/`.

## Evidence From Runs

Latest controlled comparison:

```text
python Core_SAC.py compare --eval-episodes 2 --max-episode-steps 200 --seed 7
```

Saved under `logs/20260414_115930_compare_improved_residual_pd_seed7`.

| Controller | Mean error (m) | RMS error (m) | Max error (m) | Final error (m) | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy direct torque | 0.200077 | 0.217833 | 0.449405 | 0.160011 | 0.0125 |
| Improved residual-PD | 0.050482 | 0.059722 | 0.115160 | 0.030465 | 0.1975 |

The structural fix improves untrained deterministic tracking by about `3.96x` in mean tracking error.

Short training smoke test:

```text
python Core_SAC.py train --total-steps 1200 --eval-interval 600 --eval-episodes 2 --max-episode-steps 200 --start-steps 200 --update-after 200 --batch-size 64 --seed 7
```

Saved under `logs/20260414_115941_train_improved_residual_pd_seed7`.

- Initial evaluation mean error: `0.013738 m`
- Best evaluation mean error: `0.013738 m`
- Final evaluation mean error: `0.023415 m`
- Final success rate: `0.635`

This short run verifies the training pipeline, logging, checkpointing, and evaluation path. It does not prove SAC convergence; the best checkpoint remains the initial residual-PD baseline for this short smoke test.

## Commands

Train the improved residual controller:

```text
python Core_SAC.py train --total-steps 20000 --eval-interval 5000 --eval-episodes 5 --variant improved --control-mode residual_pd
```

Evaluate a checkpoint:

```text
python Core_SAC.py eval --checkpoint logs/<run>/checkpoints/best.pt --variant improved --control-mode residual_pd --eval-episodes 5
```

Render a checkpoint with pygame:

```text
python Core_SAC.py render --checkpoint logs/<run>/checkpoints/best.pt --variant improved --control-mode residual_pd
```

Run the legacy-vs-improved ablation:

```text
python Core_SAC.py compare --eval-episodes 5 --max-episode-steps 600
```

## Remaining Work

- Run longer training to determine whether SAC residuals can beat the PD baseline consistently.
- Tune residual action limit and reward weights after longer-run evaluation, not by blind search.
- Consider evaluating PD-only explicitly as a named ablation if deeper controller comparisons are needed.
