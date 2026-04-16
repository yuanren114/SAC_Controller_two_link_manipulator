# Two-Link Arm SAC Trajectory Tracking

This project trains, evaluates, and visualizes a Soft Actor-Critic controller for a two-link planar robotic arm. The current controller uses direct joint torque actions, a meter-based tracking state, and selectable target trajectories.

Core files:

- `Core_SAC.py`: arm dynamics, trajectory generation, state construction, reward/metrics, training, evaluation, rendering, logging, and CLI commands.
- `sac_agent.py`: SAC actor, critics, replay buffer, losses, entropy tuning, LSTM layers, and checkpoint save/load.
- `logs/`: timestamped experiment outputs with configs, metrics, plots, trajectory CSV files, and checkpoints.

## Usage Cheat Sheet

Default behavior:

- Default trajectory mode is `circle`.
- Available trajectory modes are `circle` and `pdf`.
- SAC actions are direct joint torque commands bounded by the configured action limit.
- Training/evaluation outputs are saved under `logs/...`.
- Evaluation outputs and plots are saved inside the run directory.

Trajectory modes:

- `circle`: the original circular trajectory geometry, represented internally in meters for consistency with the current state pipeline.
- `pdf`: the PDF-aligned sinusoidal target trajectory projected into the current 2D arm environment.

Commands:

```powershell
python Core_SAC.py train
```

Runs training, periodic evaluation, checkpointing, CSV/JSON logging, and plot generation. It does not open real-time pygame visualization.

```powershell
python Core_SAC.py train --trajectory-mode circle
```

Runs training with the default original circular trajectory.

```powershell
python Core_SAC.py train --trajectory-mode pdf
```

Runs training with the PDF-style sinusoidal target trajectory.

```powershell
python Core_SAC.py render --checkpoint <path_to_checkpoint>
```

Opens pygame visualization. If a checkpoint is provided, the renderer loads that policy; without a checkpoint, it renders the current untrained policy.

```powershell
python Core_SAC.py train-render --trajectory-mode circle
```

Runs training first, then launches pygame visualization using the trained best checkpoint.

Output locations:

- Each run creates a folder under `logs/...`.
- `config.json`, `summary.md`, `evaluation.csv`, trajectory CSV files, and plots are saved in the run directory.
- Training runs also save `training_episodes.csv`, `training_steps.jsonl`, training curves, and checkpoints.
- Checkpoints are saved under `logs/<run>/checkpoints/`, including `best.pt` and `final.pt`.

## Current Control And State Behavior

`Core_SAC.py` defines a two-link planar arm with dynamics of the form:

```text
M(q) qdd + C(q, dq) + G(q) + damping = tau
```

The active control path is direct torque:

```text
tau = clip(actor_action, -action_limit, action_limit)
```

The actor output is not a target position, not a delta action, and not a residual added to a PD controller.

The state follows the current meter-based tracking structure:

```text
[q, dq, x, dx, x_d, dx_d, e, de]
```

where:

- `q`, `dq` are joint angle and joint velocity.
- `x`, `dx` are end-effector position and velocity in meters.
- `x_d`, `dx_d` are target position and target velocity in meters.
- `e`, `de` are position and velocity tracking errors.

## Trajectories

### `circle`

The `circle` trajectory preserves the original geometry and motion:

```text
x_px = arm.base[0] + radius_px * cos(omega * t)
y_px = arm.base[1] - center_offset_px + radius_px * sin(omega * t)
```

Internally, this is converted to meters with `arm.scale`, so the state and reward remain self-consistent.

### `pdf`

The `pdf` trajectory uses the PDF-style sinusoidal target projected into the current 2D environment:

```text
x_d = 0.1 sin(t) + 0.12
y_d = 0.1 cos(t) + 0.12
```

Both trajectory modes provide target position `x_d` and target velocity `dx_d` in meters to the same state and reward pipeline.

## Logs And Output Structure

Each run creates a timestamped folder:

```text
logs/YYYYMMDD_HHMMSS_<command>_<variant>_<control_mode>_seed<seed>/
```

Typical files:

```text
config.json
summary.md
training_episodes.csv
training_steps.jsonl
training_curves.png
evaluation.csv
evaluation_tracking_error.png
checkpoints/best.pt
checkpoints/final.pt
eval/<step_label>/episode_000_trajectory.csv
eval/<step_label>/episode_000_trajectory.png
eval/<step_label>/summary.json
```

## Other Useful Commands

Evaluate a checkpoint:

```powershell
python Core_SAC.py eval --checkpoint <path_to_checkpoint> --trajectory-mode circle
```

Run the comparison command:

```powershell
python Core_SAC.py compare --eval-episodes 5
```

Show CLI help:

```powershell
python Core_SAC.py train --help
python Core_SAC.py train-render --help
python Core_SAC.py eval --help
python Core_SAC.py render --help
python Core_SAC.py compare --help
```

## Notes

- Use `train` for learning and logging without real-time visualization.
- Use `render` for pygame visualization.
- Use `train-render` when you want one command that trains first and then visualizes the trained checkpoint.
- Use `best.pt` when you want the checkpoint with the lowest evaluation tracking error.
- Use `final.pt` when you specifically want the last policy from a training run.
