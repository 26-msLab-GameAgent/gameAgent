# References

## NitroGen

- Paper: https://arxiv.org/abs/2601.02427
- Key idea: generalist gaming agents should use a unified vision-action interface, large gameplay/action datasets, and cross-game evaluation.
- How this project uses it: keep action schemas normalized and log every frame/action pair so later training or evaluation can reuse the data.

## Voyager

- Paper: https://arxiv.org/abs/2305.16291
- Project: https://voyager.minedojo.org/
- Key idea: an LLM agent can combine curriculum, skill memory, and iterative feedback to improve over long horizons.
- How this project uses it: separate high-level goals, low-level actions, reusable skills, and episode reflection.

## ReAct

- Paper: https://arxiv.org/abs/2210.03629
- Key idea: combine reasoning traces and environment actions so the loop is inspectable and correctable.
- How this project uses it: decisions include an intent/summary plus a structured action, not only raw coordinates.

## Android / ADB Control

- Android Debug Bridge docs: https://developer.android.com/tools/adb
- How this project uses it: BlueStacks and real Android devices can both be driven through shell input commands once an ADB device is visible.

