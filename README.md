# Two-Link Arm SAC Trajectory Tracking

Train, evaluate, and visualize a Soft Actor-Critic controller for trajectory tracking on a two-link planar robotic arm.

## Overview

This project applies SAC to an end-effector trajectory tracking problem. The agent observes the arm state, target state, and tracking error, then outputs direct joint torque commands.

The current system is a two-link planar arm with:

- direct torque control
- meter-based tracking state
- selectable target trajectories
- training, evaluation, logging, plotting, and pygame visualization

Core files:

- `Core_SAC.py`: arm dynamics, trajectory generation, state construction, reward/metrics, training, evaluation, rendering, logging, and CLI commands.
- `sac_agent.py`: SAC actor, critics, replay buffer, losses, entropy tuning, and checkpoint save/load.

## Features

- Direct torque SAC control.
- `circle` and `pdf` trajectory modes.
- Training without real-time rendering for faster iteration.
- Checkpoint evaluation and best-policy tracking.
- Pygame visualization for trained or untrained policies.
- Timestamped logs, CSV metrics, trajectory plots, and training curves.

## Quick Start

Train with the default settings:

```powershell
python Core_SAC.py train
```

Train, then render the best checkpoint:

```powershell
python Core_SAC.py train-render --trajectory-mode circle
```

## Command Line Usage

### Training

```powershell
python Core_SAC.py train
```

Runs training with the default `circle` trajectory. Training saves logs, checkpoints, evaluations, and plots. It does not open pygame visualization.

### Training (custom config)

```powershell
python Core_SAC.py train `
  --trajectory-mode circle `
  --total-steps 20000 `
  --eval-interval 5000 `
  --eval-episodes 5 `
  --max-episode-steps 200 `
  --start-steps 200 `
  --update-after 200 `
  --batch-size 64
```

Use this form for faster smoke tests or longer training runs by changing the step counts and batch settings.

### Render

```powershell
python Core_SAC.py render --checkpoint <path_to_checkpoint>
```

Opens pygame visualization. When `--checkpoint` is provided, the renderer loads that trained policy.

### Train + Render

```powershell
python Core_SAC.py train-render --trajectory-mode circle
```

Runs training first, then launches pygame visualization using the trained best checkpoint.

### Evaluate

```powershell
python Core_SAC.py eval --checkpoint <path_to_checkpoint> --trajectory-mode circle
```

Runs deterministic evaluation and saves metrics, trajectory CSV files, and plots in a new log directory.

### Help

```powershell
python Core_SAC.py train --help
python Core_SAC.py train-render --help
python Core_SAC.py eval --help
python Core_SAC.py render --help
python Core_SAC.py compare --help
```

## Key Arguments

- `--trajectory-mode`
  - Selects the target trajectory.
  - `circle` is the default and preserves the original circular path.
  - `pdf` uses the PDF-style sinusoidal target projected into the current 2D arm environment.

- `--total-steps`
  - Total training steps across all episodes.
  - Increase this for better policy performance.

- `--max-episode-steps`
  - Maximum steps per episode.
  - Smaller values make experiments iterate faster.

- `--eval-interval`
  - How often evaluation runs during training.
  - Larger values reduce evaluation overhead and make training faster.

- `--eval-episodes`
  - Number of evaluation rollouts per evaluation point.
  - Smaller values are faster; larger values give more stable metrics.

- `--start-steps`
  - Number of random exploration steps before using the policy for data collection.

- `--update-after`
  - Number of collected transitions required before gradient updates begin.

- `--batch-size`
  - SAC minibatch size for gradient updates.

- `--checkpoint`
  - Path to a saved model checkpoint.
  - Used by `render` and `eval` to load a trained policy.

## Training vs Evaluation vs Rendering

- `train`
  - Runs learning.
  - Saves logs, checkpoints, evaluations, and plots.
  - Does not show real-time pygame visualization.

- `eval`
  - Loads a checkpoint if provided.
  - Runs deterministic rollouts.
  - Saves evaluation metrics and trajectory plots.

- `render`
  - Opens pygame visualization.
  - Uses a trained policy when `--checkpoint` is provided.

- `train-render`
  - Runs training first.
  - Then visualizes the best checkpoint from that training run.

## Trajectory Modes

- `circle`
  - Default mode.
  - Preserves the original circular path geometry and angular speed.
  - Internally converts the original pixel-space circle to meters for consistency with the current state and reward pipeline.

- `pdf`
  - Uses the PDF-style sinusoidal target trajectory in the current 2D environment:

```text
x_d = 0.1 sin(t) + 0.12
y_d = 0.1 cos(t) + 0.12
```

Both modes provide target position and target velocity in meters to the same state, reward, training, evaluation, and rendering code.

## Control And State

The arm dynamics use:

```text
M(q) qdd + C(q, dq) + G(q) + damping = tau
```

The active control path is direct torque:

```text
tau = clip(actor_action, -action_limit, action_limit)
```

The actor output is not a target position, delta action, or residual added to a PD controller.

The current state structure is:

```text
[q, dq, x, dx, x_d, dx_d, e, de]
```

where:

- `q`, `dq` are joint angle and joint velocity.
- `x`, `dx` are end-effector position and velocity in meters.
- `x_d`, `dx_d` are target position and target velocity in meters.
- `e`, `de` are position and velocity tracking errors.

## Outputs

Each run creates a timestamped directory under `logs/`:

```text
logs/<timestamp>_<command>_<variant>_<control_mode>_seed<seed>/
```

Common outputs:

```text
logs/<timestamp>/
  config.json
  summary.md
  training_episodes.csv
  training_steps.jsonl
  training_curves.png
  evaluation.csv
  evaluation_tracking_error.png
  checkpoints/
    best.pt
    final.pt
  eval/
    <step_label>/
      episode_000_trajectory.csv
      episode_000_trajectory.png
      summary.json
```

Where to look:

- Best checkpoint: `logs/<run>/checkpoints/best.pt`
- Final checkpoint: `logs/<run>/checkpoints/final.pt`
- Training curve: `logs/<run>/training_curves.png`
- Evaluation metrics: `logs/<run>/evaluation.csv`
- Evaluation tracking plot: `logs/<run>/evaluation_tracking_error.png`
- Per-episode trajectory plots: `logs/<run>/eval/<step_label>/episode_XXX_trajectory.png`

## Notes

- Use `best.pt` when you want the checkpoint with the lowest evaluation tracking error.
- Use `final.pt` when you specifically want the last policy from a training run.
- Use smaller `--max-episode-steps`, `--eval-episodes`, and `--batch-size` values for quick smoke tests.
- Use larger `--total-steps` for meaningful training runs.
