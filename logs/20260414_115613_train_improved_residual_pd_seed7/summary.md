# SAC Tracking Run Summary

- Command: `train`
- Variant: `improved`
- Control mode: `residual_pd`
- Seed: `7`
- Total training steps: `1200`
- Final mean tracking error: `0.029058 m`
- Final RMS tracking error: `0.032737 m`
- Final max tracking error: `0.050338 m`
- Final success rate: `0.345`
- Best evaluation mean error: `0.016930 m`

The improved controller keeps the existing IK target and PD torque as the stabilizing baseline, while SAC learns bounded residual torque. This fixes the original mismatch where the code title described SAC plus PD but the loop applied SAC torque directly.
