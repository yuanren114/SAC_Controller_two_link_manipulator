# Two-Link Arm SAC Trajectory Tracking

This project trains and evaluates a Soft Actor-Critic controller for a two-link robotic arm tracking a circular end-effector trajectory.

The current code keeps the original project structure small:

- `Core_SAC.py`: arm dynamics, trajectory target generation, observations, rewards, training loop, evaluation loop, pygame rendering, logging, and CLI commands.
- `sac_agent.py`: generic SAC implementation, including actor, critics, replay buffer, losses, entropy tuning, and checkpoint save/load.
- `logs/`: timestamped experiment outputs with configs, metrics, plots, trajectory CSV files, and checkpoints.

## What Improved Compared To The Original Version

The original version had the main pieces needed for SAC, but the control/training pipeline made trajectory tracking hard to debug and unstable.

Important improvements:

- Added reproducible CLI entry points for `train`, `eval`, `render`, `compare`, and `train-render`.
- Separated fast training/evaluation from pygame rendering, so training is not forced to run at 60 Hz.
- Added timestamped experiment folders under `logs/`.
- Added `config.json` snapshots for each run.
- Added training logs:
  - `training_episodes.csv`
  - `training_steps.jsonl`
- Added evaluation logs:
  - `evaluation.csv`
  - per-episode trajectory CSV files
  - `summary.json`
- Added plots:
  - target vs actual trajectory plots
  - tracking error plots
  - training curves
  - evaluation tracking error curves
- Added model checkpoints:
  - `checkpoints/best.pt`
  - `checkpoints/final.pt`
  - intermediate step checkpoints
- Added deterministic checkpoint evaluation.
- Added a legacy-vs-improved comparison command.
- Fixed the main controller mismatch: the old code said `SAC(delta_q) + PD`, but SAC output was actually applied directly as torque and the existing `pd_torque()` helper was unused.
- Added `residual_pd` mode, where the existing PD controller provides stable baseline tracking and SAC learns a bounded residual torque.
- Added normalized improved observations instead of mixing radians, rad/s, and raw pixels.
- Added meter-scale tracking reward and explicit tracking metrics.
- Initialized the SAC actor output near zero so the residual policy starts from the PD baseline instead of a random deterministic residual action.
- Fixed `.gitignore` so generated `__pycache__/` files are ignored.

Measured quick comparison from the included debug run:

| Controller | Mean error (m) | RMS error (m) | Max error (m) | Final error (m) | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy direct torque | 0.200077 | 0.217833 | 0.449405 | 0.160011 | 0.0125 |
| Improved residual-PD | 0.050482 | 0.059722 | 0.115160 | 0.030465 | 0.1975 |

This is about a `3.96x` mean tracking error improvement before long SAC training.

## Core_SAC.py Principles

`Core_SAC.py` defines the physical task and experiment workflow.

The arm is a two-link planar robot. Its state is:

- joint angle `q`
- joint velocity `dq`
- joint acceleration `qdd`

The dynamics use the standard manipulator equation:

```text
M(q) qdd + C(q, dq) + G(q) + damping = tau
```

where:

- `M(q)` is the mass/inertia matrix.
- `C(q, dq)` is the Coriolis/centrifugal term.
- `G(q)` is the gravity term.
- `tau` is the applied joint torque.

The target trajectory is a circle in screen coordinates. Each target end-effector point is converted to a desired joint target with inverse kinematics:

```text
target end-effector point -> q_des
next target point -> q_des_next
dq_des = (q_des_next - q_des) / dt
```

The improved controller uses:

```text
tau_total = clip(tau_PD + tau_SAC_residual, -20, 20)
```

where:

- `tau_PD` comes from the existing PD controller.
- `tau_SAC_residual` is learned by SAC.
- The residual is bounded by `--action-limit`.

This makes learning easier because the agent does not need to discover basic stabilizing control from scratch. SAC only needs to improve or compensate around a working PD baseline.

The improved observation includes normalized tracking-relevant information:

- `sin(q)`, `cos(q)` for angle representation
- normalized `dq`
- `sin(q_des)`, `cos(q_des)`
- normalized `dq_des`
- normalized joint tracking error
- normalized end-effector tracking error

The improved reward uses meter-scale quantities:

- end-effector tracking error penalty
- progress reward when error decreases
- joint tracking error penalty
- action effort penalty
- residual torque penalty
- small success bonus when tracking error is below threshold

## sac_agent.py Principles

`sac_agent.py` implements Soft Actor-Critic, an off-policy reinforcement learning algorithm for continuous actions.

SAC has three main parts:

- Actor: chooses actions.
- Critic 1 and Critic 2: estimate action value.
- Replay buffer: stores past transitions for off-policy learning.

The actor outputs a Gaussian policy:

```text
state -> mean, log_std -> sample action -> tanh -> normalized action [-1, 1]
```

The normalized action is scaled to the environment action range. In `residual_pd` mode, this means the action is residual torque.

SAC uses two critics to reduce Q-value overestimation:

```text
Q_target = min(Q1_target, Q2_target) - alpha * log_prob(action)
```

The critic loss trains Q-functions to match the Bellman target:

```text
target = reward + gamma * (1 - done) * Q_target
critic_loss = MSE(Q, target)
```

The actor loss encourages actions with high Q-value while keeping entropy:

```text
actor_loss = mean(alpha * log_prob(action) - Q(action))
```

The entropy coefficient `alpha` is automatically tuned so the policy keeps enough exploration during training.

In this project, the actor output head is initialized near zero. That matters because the improved controller is residual control: zero SAC action means the controller behaves like the PD baseline.

## Logs And Output Structure

Each run creates a timestamped folder:

```text
logs/YYYYMMDD_HHMMSS_<command>_<variant>_<control_mode>_seed<seed>/
```

Typical files:

```text
config.json
training_episodes.csv
training_steps.jsonl
evaluation.csv
training_curves.png
evaluation_tracking_error.png
summary.md
checkpoints/best.pt
checkpoints/final.pt
eval/<step_label>/episode_000_trajectory.csv
eval/<step_label>/episode_000_trajectory.png
eval/<step_label>/summary.json
```

## Command Cheatsheet

### Train Only

```powershell
python Core_SAC.py train --total-steps 20000 --eval-interval 5000 --eval-episodes 5 --variant improved --control-mode residual_pd
```

### Train And Then Automatically Render Best Checkpoint

```powershell
python Core_SAC.py train-render --total-steps 20000 --eval-interval 5000 --eval-episodes 5 --variant improved --control-mode residual_pd
```

Quick smoke-test version:

```powershell
python Core_SAC.py train-render --total-steps 1200 --eval-interval 600 --eval-episodes 2 --max-episode-steps 200 --start-steps 200 --update-after 200 --batch-size 64
```

### Render From One Existing Training Log

Use the `best.pt` or `final.pt` checkpoint inside that log folder.

Best checkpoint:

```powershell
python Core_SAC.py render --checkpoint logs\<run-folder>\checkpoints\best.pt --variant improved --control-mode residual_pd
```

Final checkpoint:

```powershell
python Core_SAC.py render --checkpoint logs\<run-folder>\checkpoints\final.pt --variant improved --control-mode residual_pd
```

Example:

```powershell
python Core_SAC.py render --checkpoint logs\20260414_115941_train_improved_residual_pd_seed7\checkpoints\best.pt --variant improved --control-mode residual_pd
```

### Evaluate One Existing Training Log

```powershell
python Core_SAC.py eval --checkpoint logs\<run-folder>\checkpoints\best.pt --variant improved --control-mode residual_pd --eval-episodes 5
```

### Compare Legacy And Improved Controllers

```powershell
python Core_SAC.py compare --eval-episodes 5 --max-episode-steps 600
```

### Show CLI Help

```powershell
python Core_SAC.py train --help
python Core_SAC.py train-render --help
python Core_SAC.py eval --help
python Core_SAC.py render --help
python Core_SAC.py compare --help
```

## Recommended Workflow

1. Run a quick smoke test:

```powershell
python Core_SAC.py train-render --total-steps 1200 --eval-interval 600 --eval-episodes 2 --max-episode-steps 200 --start-steps 200 --update-after 200 --batch-size 64
```

2. Run a longer training job:

```powershell
python Core_SAC.py train --total-steps 50000 --eval-interval 5000 --eval-episodes 5 --variant improved --control-mode residual_pd
```

3. Render the best checkpoint:

```powershell
python Core_SAC.py render --checkpoint logs\<run-folder>\checkpoints\best.pt --variant improved --control-mode residual_pd
```

4. Inspect:

- `evaluation.csv`
- `training_episodes.csv`
- `training_curves.png`
- `evaluation_tracking_error.png`
- `eval/<step_label>/*trajectory.png`

## Notes

- `render` is for visualization.
- `train` is for fast learning and logging.
- `train-render` is the one-command convenience path: train first, then show the best result.
- Use `best.pt` when you want the checkpoint with the lowest evaluation tracking error.
- Use `final.pt` when you specifically want the last policy from the training run.
