# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `residual_pd`
- Seed: `7`
- Total training steps: `20000`
- Final mean tracking error: `0.014315 m`
- Final RMS tracking error: `0.016168 m`
- Final max tracking error: `0.031046 m`
- Final success rate: `0.758`
- Best evaluation mean error: `0.009534 m`

The improved controller keeps the existing IK target and PD torque as the stabilizing baseline, while SAC learns bounded residual torque. This fixes the original mismatch where the code title described SAC plus PD but the loop applied SAC torque directly.
