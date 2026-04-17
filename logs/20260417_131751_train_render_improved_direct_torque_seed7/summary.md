# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `direct_torque`
- Seed: `7`
- Total training steps: `50000`
- Final mean tracking error: `0.034310 m`
- Final RMS tracking error: `0.050417 m`
- Final max tracking error: `0.199282 m`
- Final success rate: `0.553`
- Best evaluation mean error: `0.034310 m`

The active controller applies the SAC policy output directly as joint torque bounded by the PDF action range.
