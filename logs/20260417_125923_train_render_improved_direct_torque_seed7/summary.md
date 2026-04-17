# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `direct_torque`
- Seed: `7`
- Total training steps: `24000`
- Final mean tracking error: `0.059769 m`
- Final RMS tracking error: `0.066702 m`
- Final max tracking error: `0.199229 m`
- Final success rate: `0.067`
- Best evaluation mean error: `0.059769 m`

The active controller applies the SAC policy output directly as joint torque bounded by the PDF action range.
