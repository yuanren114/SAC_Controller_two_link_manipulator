# SAC Tracking Run Summary

- Command: `train_render`
- Variant: `improved`
- Control mode: `direct_torque`
- Seed: `7`
- Total training steps: `50000`
- Final mean tracking error: `0.005234 m`
- Final RMS tracking error: `0.017225 m`
- Final max tracking error: `0.280686 m`
- Final success rate: `0.989`
- Best evaluation mean error: `0.005234 m`

The active controller applies the SAC policy output directly as joint torque bounded by the PDF action range.
