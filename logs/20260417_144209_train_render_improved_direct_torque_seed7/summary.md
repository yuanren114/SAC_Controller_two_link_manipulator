# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `direct_torque`
- Seed: `7`
- Total training steps: `50000`
- Final mean tracking error: `0.030170 m`
- Final RMS tracking error: `0.044792 m`
- Final max tracking error: `0.148748 m`
- Final success rate: `0.445`
- Best evaluation mean error: `0.026022 m`

The active controller applies the SAC policy output directly as joint torque bounded by the PDF action range.
