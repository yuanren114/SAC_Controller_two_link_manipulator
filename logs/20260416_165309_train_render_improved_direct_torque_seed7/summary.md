# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `direct_torque`
- Seed: `7`
- Total training steps: `3000`
- Final mean tracking error: `0.154191 m`
- Final RMS tracking error: `0.159958 m`
- Final max tracking error: `0.281218 m`
- Final success rate: `0.000`
- Best evaluation mean error: `0.154191 m`

The active controller applies the SAC policy output directly as joint torque bounded by the PDF action range.
