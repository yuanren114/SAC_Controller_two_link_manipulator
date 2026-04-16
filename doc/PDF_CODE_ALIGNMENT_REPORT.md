# PDF/Code Alignment Report

## Summary of the PDF Setup

Ground-truth document: `doc/Trajectory Tracking Control for Robotic Manipulator Based on Soft Actor-Critic and Generative Adversarial Imitation Learning.pdf`.

The paper studies task-space trajectory tracking for the end effector of a Phantom Omni-style 3-DoF manipulator. The controller is learned without relying on a precise manipulator model, under joint-angle saturation and torque-input saturation.

The paper's control objective is to control joint angles by applying torques to the robot joints so the end effector tracks a time-varying task-space curve. The controller output is direct joint torque, not a target joint position, target delta, or residual correction on top of a PD controller.

Relevant setup details extracted from the PDF:

- Controlled system: 3-DoF manipulator.
- Initial joint state: `q1(0) = -0.3`, `q2(0) = 0.3`, `q3(0) = -0.8`.
- Simulation length: `3000` steps.
- Step size: `0.01 s`.
- Desired task-space trajectory:
  - `x_d = 0.1 sin(t) + 0.12`
  - `y_d = 0.1 cos(t) + 0.12`
  - `z_d = 0.1 sin(t)`
  - `t in (0, 30 s)`.
- State space, Equation (19): `s = [q, dq, x, dx, x_d, dx_d, e, de]^T`.
  - `x` and `dx` are end-effector position and velocity.
  - `x_d` and `dx_d` are target end-effector position and velocity.
  - `e = x - x_d` is tracking error.
  - `de` is speed error.
- Joint-angle saturation, Equation (20): `q_i` is clipped to `[q_min, q_max]`, with `q_min = -pi`, `q_max = pi`.
- Action space, Equations (21)-(22): `a = [tau]^T`, where each torque component is clipped to `[tau_min, tau_max]`, with `tau_min = -5`, `tau_max = 5`.
- Reward, Equation (23): nonlinear tracking-error reward using `V = 2` and error threshold `e = 0.02`.
- SAC setup:
  - Maximum entropy SAC objective.
  - Twin Q networks plus target Q networks.
  - Policy outputs Gaussian action parameters; sampled action is torque.
  - LSTM is added to SAC. The actor receives state sequences, feeds them through an LSTM, then through fully connected layers, then outputs Gaussian parameters for torque.
  - Critic uses state sequences concatenated with actions.
- GAIL setup:
  - SAC-LSTM policy is used as the generator.
  - SAC-LSTM policies provide expert demonstrations.
  - Discriminator receives state-action pairs.
  - GAIL surrogate reward: `r(s, a) = -log(D(s, a))`.
  - The final proposed method is SL-GAIL, not plain SAC.
- Algorithm parameters from Appendix A:
  - actor learning rate: `0.0003`
  - critic learning rate: `0.003`
  - discriminator learning rate: `0.001`
  - alpha learning rate: `0.0003`
  - soft update parameter: `0.005`
  - replay buffer size: `1,000,000`
  - minimal size: `5000`
  - batch size: `256`
  - LSTM hidden size: `128`
  - LSTM num layers: `1`

## Summary of the Current Code

`Core_SAC.py` implements a 2-DoF planar two-link arm, not the PDF's 3-DoF Phantom Omni manipulator. It includes arm dynamics, inverse kinematics, forward kinematics, a PD torque helper, circular target generation, state construction, reward metrics, training/evaluation/rendering, and CLI commands.

`sac_agent.py` implements generic SAC with:

- MLP actor.
- MLP twin critics.
- target critics.
- replay buffer.
- tanh-squashed Gaussian actions scaled to an environment action range.
- automatic entropy tuning.
- checkpoint save/load.

Current high-level behavior:

- Default mode is `variant="improved"` and `control_mode="residual_pd"`.
- The actor outputs a bounded residual torque in `[-action_limit, action_limit]`.
- `Core_SAC.control()` computes `tau_pd = pd_torque(q_des, dq_des)`, adds the SAC residual torque, then clips total torque to `[-20, 20]`.
- The current target is a 2D screen-space circle defined by `radius_px`, `center_offset_px`, and `omega`.
- The improved observation is a custom normalized vector containing trigonometric joint features, desired joint features, joint errors, and end-effector error.
- The improved reward is a custom meter-scale reward with tracking, progress, joint-error, action-effort, residual-torque, and success terms.
- Default simulation step is `1/60 s`.
- Default episode length is `600`.
- Default replay size is `100000`.
- Default batch size is `128`.

## Mismatch Analysis

### Manipulator Model

PDF says:

- Use a 3-DoF Phantom Omni manipulator with the dynamics/kinematics described in Section 2 and Appendix A.

Current code does:

- Uses a 2-DoF planar arm with different link lengths, masses, dynamics, and task space.

Impact:

- This is a structural mismatch. A full conversion to the PDF manipulator is larger than a SAC fidelity fix and would require replacing the plant model, kinematics, state dimension, action dimension, plotting, and rendering.

Ambiguity:

- The user asked specifically to make the SAC implementation match the PDF. I will not invent a full 3-DoF Phantom Omni implementation beyond the available code without explicit confirmation. I will make the SAC/control semantics match the PDF as far as the existing 2-DoF environment allows, and record the remaining 3-DoF plant mismatch.

### Action Semantics

PDF says:

- The controller output signals are torques acting on manipulator joints.
- Action is `a = [tau]^T`.
- Torque is saturated to `[-5, 5]`.
- No PD target, delta-position command, or residual-PD baseline is described.

Current code does:

- Default action is residual torque in `residual_pd` mode.
- Adds actor output to `pd_torque(q_des, dq_des)`.
- Clips total torque to `[-20, 20]`.
- Still exposes both `direct_torque` and `residual_pd` modes.

Required change:

- Make SAC action direct applied joint torque.
- Set action bounds to `[-5, 5]`.
- Remove residual-PD behavior from the default and control path.
- Keep `pd_torque()` only as unused legacy code or remove it if cleaning further; it must not be in the active SAC control path.

### State Space

PDF says:

- State is `[q, dq, x, dx, x_d, dx_d, e, de]^T`.

Current code does:

- Legacy state: `[q, dq, q_des, dq_des, ee_px, target_px]`.
- Improved state: normalized sin/cos joint features, desired joint features, joint errors, and end-effector error.

Required change:

- Construct the state from actual joint state, end-effector position/velocity, target position/velocity, position error, and velocity error.
- For the current 2-DoF arm, the closest faithful state is 16-dimensional:
  - `q(2), dq(2), x(2), dx(2), x_d(2), dx_d(2), e(2), de(2)`.
- For full PDF fidelity this should be 24-dimensional for the 3-DoF task-space case:
  - `q(3), dq(3), x(3), dx(3), x_d(3), dx_d(3), e(3), de(3)`.

### Target Trajectory

PDF says:

- Desired task-space trajectory is sinusoidal:
  - `x_d = 0.1 sin(t) + 0.12`
  - `y_d = 0.1 cos(t) + 0.12`
  - `z_d = 0.1 sin(t)`.

Current code does:

- Uses a pixel-space circle centered at the screen base with radius and angular rate.

Required change:

- For the current 2-DoF planar environment, use the PDF's `x_d` and `y_d` components in meters and map them to screen coordinates only for rendering/logging.
- Record that `z_d` cannot be represented until the plant is converted to 3-DoF.

### Reward

PDF says:

- Uses nonlinear tracking-error reward from Equation (23), with `V = 2` and threshold `0.02`.

Current code does:

- Uses custom reward terms: tracking error, progress, joint error, action effort, residual torque effort, success bonus.

Required change:

- Replace the improved reward with the PDF nonlinear tracking-error reward.
- Remove residual/action-effort terms unless needed only for logging; they are not part of the PDF reward.

### SAC Network

PDF says:

- SAC-LSTM actor: state sequences go through LSTM, then fully connected layers, then Gaussian action parameters.
- LSTM hidden size is `128`, layer count is `1`.
- Critic takes state sequences and actions.

Current code does:

- MLP actor and MLP critics only.
- No sequence input or LSTM.

Required change:

- Add LSTM-based actor/critic support or replace the MLP networks with LSTM variants.
- Minimal safe step: add LSTM layers that accept both `(batch, state_dim)` and `(batch, seq_len, state_dim)` and treat single states as length-1 sequences. This aligns network structure but does not fully implement replayed state sequences.
- Full PDF fidelity would require replaying state sequences, not isolated one-step states.

### GAIL / SL-GAIL

PDF says:

- Final proposed method is SL-GAIL: SAC-LSTM generator plus GAIL discriminator and surrogate reward.

Current code does:

- Plain SAC only.
- No discriminator, expert data, GAIL reward, or imitation training loop.

Required change:

- Full PDF fidelity requires implementing GAIL and expert-data generation.

Ambiguity:

- The repo currently contains SAC-related code only, and the user specifically called out the SAC implementation and action semantics. I will not add a full GAIL subsystem in this pass because that would be a large new algorithmic component rather than a minimal SAC fidelity refactor.

### Training / Simulation Parameters

PDF says:

- `dt = 0.01`
- `total steps = 3000`
- batch size `256`
- replay buffer `1,000,000`
- minimum replay size `5000`
- actor LR `0.0003`
- critic LR `0.003`
- alpha LR `0.0003`
- soft update `0.005`

Current code does:

- `dt = 1/60`
- `total_steps = 20000`
- `max_episode_steps = 600`
- batch size `128`
- replay buffer `100000`
- update-after/start-steps `1000`
- critic LR `0.0003`

Required change:

- Update defaults to the PDF values where the current code has equivalent parameters.

## Refactor Plan

1. Change active control semantics in `Core_SAC.py`:
   - Default `control_mode` becomes `direct_torque`.
   - CLI default becomes `direct_torque`.
   - `make_agent()` always uses action bounds `[-5, 5]`.
   - `control()` applies actor output directly as torque and clips to `[-5, 5]`.
   - Remove residual-PD addition from active behavior.

2. Change state construction in `Core_SAC.py`:
   - Add end-effector velocity calculation from the planar Jacobian.
   - Add target velocity in meters from the PDF trajectory.
   - Return `[q, dq, x, dx, xd, dxd, e, de]` for the active state.
   - Set active state dimension to `16` for the current 2-DoF implementation.

3. Change target generation in `Core_SAC.py`:
   - Use PDF `x_d/y_d` trajectory in meters.
   - Convert only for visualization/logging to pixels.
   - Continue deriving `q_des` by IK only where useful for reset/logging; do not use it to define the action.

4. Change reward in `Core_SAC.py`:
   - Use Equation (23)-style nonlinear tracking-error reward with `V = 2` and threshold `0.02`.
   - Keep metrics for tracking error, torque norm, and saturation.

5. Change defaults in `Core_SAC.py`:
   - `dt = 0.01`
   - `total_steps = 3000`
   - `max_episode_steps = 3000`
   - `batch_size = 256`
   - `replay_size = 1000000`
   - `start_steps/update_after = 5000`
   - `action_limit = 5.0`

6. Change SAC network defaults in `sac_agent.py`:
   - Add LSTM to actor and critics with hidden size `128`, one layer.
   - Preserve compatibility with existing one-step replay by treating 2D tensors as one-step sequences.
   - Set critic LR default to `0.003`.
   - Keep actor LR, alpha LR, and soft update already matching the PDF.

7. Update report after implementation:
   - Add a "Changes Implemented" section.
   - State remaining ambiguities: 2-DoF environment and no GAIL subsystem.

## Changes Implemented

Implemented after the pre-change report above was written.

### `Core_SAC.py`

- Changed active/default control mode to `direct_torque`.
- Restricted the CLI `--control-mode` choices to `direct_torque`.
- Changed the active SAC action range to `[-5, 5]`.
- Changed `TwoLink.step()` to clip applied torque to `[-5, 5]`.
- Changed joint clipping to `[-pi, pi]` to match the PDF joint-angle saturation.
- Removed residual-PD addition from the active control path:
  - `control()` no longer calls `pd_torque()`.
  - SAC action is now the applied joint torque.
  - `tau_pd` is kept as zeros only for existing logging schema compatibility.
- Replaced the previous screen-space circular target with the PDF's sinusoidal task-space target, projected onto the existing 2D environment:
  - `x_d = 0.1 sin(t) + 0.12`
  - `y_d = 0.1 cos(t) + 0.12`
  - The PDF's `z_d` component remains unavailable because the current plant is 2-DoF.
- Replaced the custom normalized observation with the PDF-style state:
  - `q, dq, x, dx, x_d, dx_d, e, de`
  - Active dimension is `16` for the current 2-DoF arm.
- Added planar end-effector velocity calculation from the Jacobian.
- Replaced the custom reward with the PDF nonlinear tracking-error reward form:
  - `r = 1 - exp(2 * (tracking_error - 0.02))`
- Changed default run parameters toward the PDF:
  - `dt = 0.01`
  - `total_steps = 3000`
  - `max_episode_steps = 3000`
  - `batch_size = 256`
  - `replay_size = 1000000`
  - `start_steps = 5000`
  - `update_after = 5000`
  - `action_limit = 5.0`
- Updated generated run summaries and comparison report text so they no longer describe residual-PD as the active controller.

### `sac_agent.py`

- Added an LSTM layer to the actor with:
  - hidden size `128`
  - number of layers `1`
- Added an LSTM layer to each critic with:
  - hidden size `128`
  - number of layers `1`
- Preserved compatibility with the existing replay buffer by treating a 2D state tensor as a one-step sequence.
- Changed the default critic learning rate to `0.003`, matching Appendix A.
- Left actor learning rate, alpha learning rate, and soft target update unchanged because they already matched the PDF values.

## Remaining Fidelity Gaps / Ambiguities

- The environment is still the existing 2-DoF planar arm. The PDF uses a 3-DoF Phantom Omni manipulator. Matching that exactly requires a plant/kinematics replacement, not only a SAC refactor.
- The state is PDF-structured but 2D because the environment is 2-DoF. Full PDF state would be 24-dimensional for the 3-DoF task-space problem.
- The LSTM is structurally present, but the replay buffer still stores one-step states. Full SAC-LSTM fidelity would require sequence replay and training batches of state histories.
- GAIL/SL-GAIL is not implemented. The PDF's final proposed method uses SAC-LSTM expert generation plus GAIL; this repository remains SAC-LSTM-style direct RL after this pass.
- The PDF's exact 3D trajectory includes `z_d = 0.1 sin(t)`. The current code can only use the `x_d/y_d` components.

## Verification

- `python -m py_compile Core_SAC.py sac_agent.py` passed.
- A short smoke evaluation passed with `KMP_DUPLICATE_LIB_OK=TRUE` due to a local duplicate OpenMP runtime issue:
  - `python Core_SAC.py eval --eval-episodes 1 --max-episode-steps 5`
  - The smoke output confirmed the evaluation loop runs with `eval_improved_direct_torque`.
- A direct control assertion confirmed:
  - active state shape is `(16,)`
  - a dummy actor output `[9, -9]` is clipped to applied torque `[5, -5]`
  - `tau_pd` remains `[0, 0]`, confirming no active PD contribution.
