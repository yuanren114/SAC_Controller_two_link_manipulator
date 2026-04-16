# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `residual_pd`
- Seed: `7`
- Total training steps: `20000`
- Final mean tracking error: `0.011176 m`
- Final RMS tracking error: `0.013387 m`
- Final max tracking error: `0.032246 m`
- Final success rate: `0.895`
- Best evaluation mean error: `0.010547 m`

The improved controller keeps the existing IK target and PD torque as the stabilizing baseline, while SAC learns bounded residual torque. This fixes the original mismatch where the code title described SAC plus PD but the loop applied SAC torque directly.
