# NLHE 6-Max Poker Engine

Version: 0.1.0 | License: MIT | Status: Beta

Deterministic 6-max No-Limit Texas Hold'em engine with modular Python and Rust components, Gym environments, demos, and training utilities for reinforcement-learning research or production simulators.

## 1. Architectural Overview
The engine models full 6-max NLHE with integer-encoded cards (`0..51`). It enforces betting rules, generates legal actions, progresses rounds, and resolves showdowns with side-pot support. Performance-critical paths can be offloaded to Rust extensions.

## 2. Repository Layout
```
nlhe/
  core/          # Engine, datatypes, card utils, hand evaluator
  agents/        # Agent protocol and implementations
  envs/          # Gym and parameterized environments
  demo/          # CLI demos
  nlhe_eval/     # Rust hand evaluator (PyO3)
  rs_engine/     # Rust engine backend (PyO3)
  train/         # RLlib training scripts
```
docs/            # MkDocs site sources
API_SPEC.md      # Formal API specification

## 3. Core Engine (`nlhe.core`)
### 3.1 Cards (`cards.py`)
- `rank_of(c:int) -> int` returns rank `2..14`
- `suit_of(c:int) -> int` returns suit `0..3`
- `make_deck() -> List[int]` creates ordered 52-card deck

### 3.2 Datatypes (`types.py`)
- **ActionType** enum: `FOLD`, `CHECK`, `CALL`, `RAISE_TO`
- **Action** dataclass: `{kind, amount}`
- **PlayerState**: `hole`, `stack`, `bet`, `cont`, `status`, `rho`
- **GameState**: table-wide state (`button`, `round_label`, `board`, `players`, `current_bet`, `min_raise`, `tau`, `next_to_act`, `pot`, blinds, `actions_log`)
- **LegalActionInfo**: legal actions list with `min_raise_to`, `max_raise_to`, `has_raise_right`

### 3.3 Hand Evaluation (`eval.py`)
- Python fallback `best5_rank_from_7`
- Uses Rust `nlhe_eval` extension when installed for speed

### 3.4 Engine (`engine.py`)
- `reset_hand(sb:int=1, bb:int=2, start_stack:int=100, num_players:int=6, rng=None)` → initializes `GameState`
- `owed(state, seat) -> int` chips needed to call
- `legal_actions(state) -> LegalActionInfo` with raise bounds
- `step(state, action) -> (state, done, rewards, info)` applies action and advances play
- `advance_round_if_needed(state) -> (done, rewards)` deals next street or settles showdown

### 3.5 Rust Engine (`rs_engine.py`)
- Drop-in replacement backed by `nlhe_engine` PyO3 module
- Applies state diffs in place and caches `Action` objects for speed

## 4. Agents (`nlhe.agents`)
### 4.1 Protocols (`base.py`)
- **Agent**: `act(env, state, seat) -> Action`
- **EngineLike**: exposes `legal_actions` and `owed`

### 4.2 Implementations
- **TamedRandomAgent** (`tamed_random.py`): stochastic policy with configurable raise/all-in probabilities and cap
- **HumanAgent** (`human_cli.py`): CLI-driven manual play

## 5. Environment Wrappers (`nlhe.envs`)
### 5.1 `NLHEGymEnv` (`gym_env.py`)
- Observation space: hero hole, board, stacks, bets, conts, status, button, next seat, round, current bet, min raise, action mask
- Action space: `Discrete(7)` → fold, check, call, raise to `{min, min+2bb, min+4bb, all-in}`
- Non-hero seats auto-play using `TamedRandomAgent`

### 5.2 `NLHEParamEnv` (`param_env.py`)
- Action space: `Dict{atype:Discrete(4), r:Box[0,1]}` continuous raise control
- Observation: pot, current bet, padded board, hero stack/bet/cont/hole, last `H` actions
- Supports standalone use and Gym subclass

## 6. Demos (`nlhe.demo`)
- `cli.py`: interactive hand in terminal with ASCII rendering
- `envdemo.py`: demonstration of `NLHEParamEnv`

## 7. Training Utilities (`nlhe.train`)
- `PPOv2.py`: Hydra-configured RLlib PPO training
- `callbacks.py`: evaluation, checkpointing, config archival
- `loggers.py`: Slim TensorBoard logger
- `Evaluators.py`: evaluation harness
- `trainRlibPPO.py`: extended training script with optimizer patches and checkpoint management

## 8. Rust Extensions
- `nlhe_eval`: Rust hand evaluator exposed via `best5_rank_from_7_py`
- `rs_engine`: Rust state machine for fast engine execution

## 9. Data Model & Semantics
- **Card encoding**: Spades `0..12`, Hearts `13..25`, Diamonds `26..38`, Clubs `39..51`
- **Action log schema**: `(seat, action_id, amount, round_id)`; `amount` for `RAISE_TO` is new `current_bet`
- **Showdown**: side pots via `levels` algorithm; rewards are net winnings with zero-sum guarantee
- **Determinism**: engine output determined by initial seed, state, and action sequence

## 10. Example Usage
### 10.1 Engine Loop
```python
from nlhe.core.engine import NLHEngine
from nlhe.core.types import Action, ActionType

eng = NLHEngine(sb=1, bb=2, start_stack=100)
s = eng.reset_hand(button=0)
done = False
while not done:
    info = eng.legal_actions(s)
    a = Action(ActionType.CHECK) if any(x.kind == ActionType.CHECK for x in info.actions) else Action(ActionType.FOLD)
    s, done, rewards, _ = eng.step(s, a)
    if s.next_to_act is None and not done:
        done, rewards = eng.advance_round_if_needed(s)
print("Final rewards:", rewards)
```

### 10.2 Gym Environment
```python
from nlhe.envs.gym_env import NLHEGymEnv

env = NLHEGymEnv(hero_seat=0, seed=42)
obs, info = env.reset()
terminated = False
while not terminated:
    action = int(obs["action_mask"].argmax())
    obs, reward, terminated, truncated, info = env.step(action)
print("Hero reward:", reward, info.get("rewards_all"))
```

### 10.3 Parameterized Environment
```python
from nlhe.envs.param_env import NLHEParamEnv

env = NLHEParamEnv(hero_seat=0, seed=42)
obs, info = env.reset()
done = False
while not done:
    act = {"atype": 1, "r": 0.0}
    obs, r, done, _, _ = env.step(act)
print("Reward:", r)
```

## 11. Build & Dependencies
- Python 3.11+
- Gymnasium, Ray RLlib, PyTorch, Hydra
- Rust toolchain for `nlhe_eval` and `rs_engine`

Install in editable mode (builds Rust extensions if toolchain present):
```bash
pip install -e .
```

Alternatively, compile the Rust extension crates directly with the cross-platform helper:
```bash
python build_rust.py --use-maturin --crate-dir nlhe/nlhe_eval
python build_rust.py --use-maturin --crate-dir nlhe/rs_engine

```
These build the `nlhe_eval` and `rs_engine` crates and install the resulting modules into the active Python environment.

## 12. Testing & QA Recommendations
- Verify invariants: `pot == sum(cont)`, `current_bet == max(bet)` after each action
- Determinism: replay action logs with fixed seed
- Showdown: `sum(rewards) == 0`
- Fuzz random policy vs known pot allocator

## 13. Extension & Integration Points
- Implement custom agents via `Agent.act`
- Build alternative environments using `EngineLike`
- Swap hand evaluator or integrate external services
- Consume `actions_log` for analytics or training data

## License
MIT
