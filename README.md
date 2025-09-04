# NLHE 6-Max — Modular Refactor

- `nlhe/core`: pure engine + types + cards/eval (no agents, no demos)
- `nlhe/agents`: agent interfaces and implementations
- `nlhe/envs`: Gym & parameterized env wrappers
- `nlhe/demo/cli.py`: minimal interactive CLI demo

Quick test in Python:

```python
from nlhe.demo.cli import run_hand_cli
run_hand_cli()
```
