# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `direct_torque`
- Seed: `7`
- Total training steps: `24000`
- Final mean tracking error: `0.101137 m`
- Final RMS tracking error: `0.114266 m`
- Final max tracking error: `0.214415 m`
- Final success rate: `0.000`
- Best evaluation mean error: `0.070539 m`

The active controller applies the SAC policy output directly as joint torque bounded by the PDF action range.
