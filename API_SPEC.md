
# NLHE 6‑Max Engine — Industrial‑Grade API Specification
Version: 0.1.0 • Status: Beta • License: MIT

This document defines a formal, implementation‑agnostic API for the modular No‑Limit Texas Hold’em (6‑Max) engine and wrappers in `nlhe_refactor/nlhe`. It specifies interfaces, semantics, invariants, and error handling to support production use in simulators, RL systems, and tools.

---

## 1. Scope & Non‑Goals
**Scope.** Poker engine (state machine), legal action generation, betting rules (table stakes), runout & showdown with carry‑down side‑pot allocation, basic agents (for demo/testing), and two env wrappers (Gym‑style & parameterized).  
**Non‑Goals.** GUI rendering, networking, bankroll management, multi‑table tournament logic, hand history file formats.

---

## 2. Terminology & Notation
- Players indexed by seat `i ∈ {0,..,5}` clockwise; `button ∈ {0,..,5}`.
- Blinds: `sb`, `bb` (integers ≥1). Table‑stakes stacks and currency are integer chips.
- **State** `s` aggregates public board, per‑player fields, betting round label, and control variables.
- Action types: `FOLD`, `CHECK`, `CALL`, `RAISE_TO(T)` (T = total bet this round after action).
- **Owed** for seat `i`: `owed(i) = max(0, current_bet - bet[i])`.
- **Raise rights**: `rho[i]` (last response step of seat `i`) and round token `tau` implement reopening (see §6.4).

---

## 3. Determinism, RNG, and Reproducibility
- The engine is deterministic given: initial `GameState`, action sequence, and RNG (Python `random.Random` passed to `NLHEngine`).  
- **Seeding**: pass `random.Random(seed)` to the engine/agents/envs to reproduce shuffles and stochastic agent choices.
- No global RNG usage is allowed in core engine.

---

## 4. Package Layout (Public Surface)
```
nlhe/
  core/
    cards.py      # ranks/suits/deck helpers
    eval.py       # hand evaluator (Rust optional), Python fallback
    types.py      # dataclasses & enums exposed
    engine.py     # NLHEngine (the only state machine)
  agents/
    base.py       # Protocols: Agent, EngineLike
    tamed_random.py
    human_cli.py  # demo-only
  envs/
    gym_env.py    # NLHEGymEnv (Discrete action space)
    param_env.py  # NLHEParamEnv (atype + continuous r)
  demo/
    cli.py        # run_hand_cli() interactive demo
```
**Public API modules:** `nlhe.core.*`, `nlhe.agents.base`, `nlhe.agents.tamed_random`, `nlhe.envs.*`, `nlhe.demo.cli`.

---

## 5. Data Types (nlhe.core.types)
### 5.1 `ActionType` (Enum)
- `FOLD`, `CHECK`, `CALL`, `RAISE_TO`.

### 5.2 `Action`
```
@dataclass
class Action:
    kind: ActionType
    amount: Optional[int] = None  # required iff kind is RAISE_TO
```
**Preconditions:**  
- For `RAISE_TO`, `amount` must be an integer; legality is validated in `engine.step`.

### 5.3 `PlayerState`
```
@dataclass
class PlayerState:
    hole: Optional[Tuple[int,int]]  # 2 cards, ints in [0,51] or None after muck? (not used)
    stack: int                      # remaining stack >= 0
    bet: int                        # bet in current round >= 0
    cont: int                       # total hand contribution >= 0
    status: str                     # 'active' | 'folded' | 'allin'
    rho: int                        # last response step index (see §6.4)
```
**Invariants:** `stack + bet + (past paid) == initial stack` per hand. During a hand: `stack >= 0`, `bet >= 0`, `cont >= bet`.

### 5.4 `GameState`
```
@dataclass
class GameState:
    button: int              # seat id in [0,5]
    round_label: str         # 'Preflop' | 'Flop' | 'Turn' | 'River' | 'Showdown'
    board: List[int]         # dealt cards (0..51), length in {0,3,4,5}
    undealt: List[int]       # remaining deck top at end
    players: List[PlayerState]  # length 6
    current_bet: int         # max(p.bet)
    min_raise: int           # last full-raise increment this round (>= bb for opening)
    tau: int                 # round reopening token (step index)
    next_to_act: Optional[int]  # None if action round closed
    step_idx: int            # monotone action step counter (starts at 0, increments per action)
    pot: int                 # sum(p.cont)
    sb: int; bb: int
    actions_log: List[Tuple[int,int,int,int]]  # (seat, action_id, amount, round_id)
```
**Invariants (hard):**  
- `current_bet == max(p.bet)`  
- `pot == sum(p.cont)`  
- `sum(final_rewards) == 0` (zero‑sum, see §7).

### 5.5 `LegalActionInfo`
```
@dataclass
class LegalActionInfo:
    actions: List[Action]
    min_raise_to: Optional[int] = None
    max_raise_to: Optional[int] = None
    has_raise_right: Optional[bool] = None
```
For `RAISE_TO` to be present in `actions`, fields are populated as per §6.3.

---

## 6. Engine API (nlhe.core.engine.NLHEngine)
### 6.1 Constructor
```
NLHEngine(sb: int = 1, bb: int = 2, start_stack: int = 100,
          num_players: int = 6, rng: Optional[random.Random] = None)
```
**Preconditions:** `num_players == 6`, `sb, bb, start_stack` are positive integers.  
**Post:** Stateless aside from RNG; all state lives in returned `GameState` objects.

### 6.2 `reset_hand(button: int = 0) -> GameState`
- Shuffles 52‑card deck, deals 2 to each seat (top from end), posts **one** SB/BB (table stakes), sets `current_bet=bb`, `min_raise=bb`, sets `tau=0`, and `next_to_act = (button+3) % 6` (UTG preflop).  
- **Sanity:** asserts `pot == sb + bb` and `current_bet == bb`.

### 6.3 `legal_actions(s) -> LegalActionInfo`
Let `i = s.next_to_act`. Returns legal `Action` set for seat `i` and, if `RAISE_TO` is legal:
- `min_raise_to` =  
  - if `current_bet == 0`: `max(min_raise, 1)` (opening bet minimum)  
  - else: `current_bet + min_raise` (full‑raise threshold)
- `max_raise_to` = `players[i].bet + players[i].stack` (all‑in cap)  
- `has_raise_right` = `players[i].rho < s.tau` **or** `current_bet == 0`  
**Presence conditions:** if `max_raise_to > current_bet`, `RAISE_TO` is included.  
**Passive actions:**  
- `FOLD` present iff `owed(i) > 0`.  
- `CHECK` present iff `owed(i) == 0`.  
- `CALL` present iff `owed(i) > 0`.

### 6.4 Raise Rights Reopening (ρ/τ Model)
- `s.step_idx` increments per action.
- `p.rho` updated upon each response from seat `i`.
- A **full raise** (increment `raise_to - previous_current_bet >= s.min_raise`) sets `s.tau = s.step_idx`, updates `s.min_raise` to that increment, and resets `rho` of other active seats to `-∞` (rights reopened).  
- **Short all‑in**: does **not** reopen rights.  
- A seat has raise right iff `p.rho < s.tau` (hasn’t responded since last full raise) or `current_bet == 0` (opening state).

### 6.5 `owed(s, i) -> int`
Returns `max(0, current_bet - players[i].bet)`.

### 6.6 `step(s, a: Action) -> (s, done: bool, rewards: Optional[List[int]], info: dict)`
Applies exactly one action for `s.next_to_act`.  
**Validation:**  
- `CHECK` requires `owed == 0`; `CALL` requires `owed > 0`;  
- `RAISE_TO(T)` requires `T > current_bet` and `T <= players[i].bet + players[i].stack`.  
- If no raise right (`has_raise_right == False`), only `T == max_to` (all‑in) is allowed.  
- If rights open, `T >= min_raise_to` or `T == max_to`.  
- Transitions `status -> 'allin'` if `stack` hits 0 by action.  
- Updates `current_bet`, `pot`, `min_raise`, `tau`, `rho` as per rules.  
- Appends to `actions_log`: `(seat, action_id, amount, round_id)` where `amount` for `RAISE_TO` equals the new `current_bet` at the moment of action; for `CALL`, `amount` is informational (0 in current implementation; contributions are tracked by `cont`).

**Round advance:** After the action, the engine computes whether the action round is open. If closed:
- If all‑in early and not on river: auto‑deal to river and showdown.  
- Else deal next street and reset round variables (`bet=0`, `current_bet=0`, `min_raise=bb`, `tau=0`, `next_to_act = first active left of button`).

### 6.7 `advance_round_if_needed(s) -> (done: bool, rewards: Optional[List[int]])`
Idempotent helper to advance street or settle if current round is closed when `next_to_act is None`.  
**Showdown:** see §7.

**Time Complexity:** All operations are O(N) with `N=6`; showdown ranking uses evaluator once per survivor.

**Thread Safety:** `GameState` is mutable and not thread‑safe; do not share across threads without external synchronization.

---

## 7. Showdown & Side‑Pot Allocation (Carry‑Down)
**Inputs:** contributions `cont[i]` (all seats, including folded), survivors `A` = non‑folded seats, ranks for `A`.  
**Levels:** sort unique positive contribution levels `y₁<…<yₖ`.  
**Tranche Pot:** For tranche `t` covering `(y_{t-1}, y_t]`, total chips `Pk = (#players with cont ≥ y_t) * (y_t - y_{t-1}) + carry` from previous empty tranche.  
**Eligibility:** Winners of tranche `t` are survivors with `cont ≥ y_t`. If none, entire `Pk` carries down to next tranche.  
**Split:** `Pk` split equally among winners; remainder chips distributed starting from `(button+1)` clockwise.  
**Final:** Sum tranche allocations → per-seat `rewards_abs[i]`. Net rewards returned to caller are `RL[i] = rewards_abs[i] − cont[i]`.  
**Invariants:** `∑_i rewards_abs[i] == pot` and `∑_i RL[i] == 0` (asserted).

**Correctness Sketch:** Carry‑down ensures chips contributed by folded players at higher levels are not lost: if no eligible survivor at a level, chips are deferred to deeper levels where winners exist, preserving conservation and impartiality to fold timing.

---

## 8. Hand Evaluation (nlhe.core.eval)
- Card IDs: `0..51`; `rank_of(c) ∈ {2..14}`, `suit_of(c) ∈ {0..3}`.  
- `best5_rank_from_7(cards7: Iterable[int]) -> (category:int, tiebreak:Tuple[int,...])` with lexicographic ordering, higher is better.  
- If `nlhe_eval` (Rust) present, it’s used; else Python fallback enumerates C(7,5) combos.

---

## 9. Agents
### 9.1 Protocols (nlhe.agents.base)
```
class EngineLike(Protocol):
    def legal_actions(self, s: GameState) -> LegalActionInfo: ...
    def owed(self, s: GameState, i: int) -> int: ...

class Agent(Protocol):
    def act(self, env: EngineLike, s: GameState, seat: int) -> Action: ...
```
### 9.2 `TamedRandomAgent`
- Stochastic policy with caps to avoid pathological all‑ins; uses `LegalActionInfo` to pick passive/raise lines robustly in short‑stack or rights‑closed states.
- **Determinism:** reproducible given seeded RNG.

### 9.3 `HumanAgent` (demo only)
- Reads from stdin; validates inputs against `LegalActionInfo`.

---

## 10. Environment Wrappers
### 10.1 `NLHEGymEnv` (nlhe.envs.gym_env)
**Action space:** `Discrete(7)`
```
0 FOLD        (only if owe>0)
1 CHECK       (only if owe==0)
2 CALL        (only if owe>0)
3 RAISE_MIN              -> target = min_raise_to
4 RAISE_MIN_PLUS_2BB     -> target = min(min_to + 2*bb, max_to)
5 RAISE_MIN_PLUS_4BB     -> target = min(min_to + 4*bb, max_to)
6 ALLIN                  -> target = max_to
```
**Observation space:** `spaces.Dict` with keys
`hero_hole(2), board(5 padded -1), stacks(6), bets(6), conts(6), status(6), button, next_to_act, round, current_bet(1), min_raise(1), action_mask(7)`.

**Rewards:** 0 until terminal; at hand end, reward = hero’s `RL`.  
**Reset/Step Semantics:** After reset, bots auto‑act until hero’s turn or terminal. On step, apply hero action; then bots auto‑act again; if action round closes, environment calls `advance_round_if_needed`.

**Info:** On terminal, `info["rewards_all"]` contains all seats’ rewards.  
**Render:** returns a string (no side effects).

### 10.2 `NLHEParamEnv` (nlhe.envs.param_env)
**Action space:** `Dict({'atype': Discrete(4), 'r': Box([0,1])})` with mapping:  
`atype=0/1/2` → `FOLD/CHECK/CALL`; `atype=3` → `RAISE_TO` with `target = clamp(min_to + r*(max_to-min_to))`, respecting raise rights (`!has_rr` or `max_to<min_to` ⇒ forced all‑in).  
**Observation:** compact public/hero view + last `H` actions (`actions_log`), padded with `(-1,-1,-1,-1)`.

---

## 11. Action Log & Stable IDs
`GameState.actions_log: List[Tuple[int,int,int,int]]`:
- `seat ∈ [0,5]`  
- `action_id ∈ {0:Fold,1:Check,2:Call,3:RaiseTo}`  
- `amount`: for `RaiseTo` equals post‑action `current_bet`; for others implementation-specific (0).  
- `round_id ∈ {0:Preflop,1:Flop,2:Turn,3:River}` (`Showdown` recorded as 3 for compactness).

**Compatibility:** This tuple layout is considered stable for logging & offline analytics.

---

## 12. Errors & Exceptions
- `AssertionError` indicates violated invariants or misuse (e.g., calling `CHECK` when `owed>0`).  
- `ValueError` for malformed `Action` (e.g., missing `amount` for `RAISE_TO`).  
- Engine methods never raise on normal legal usage; preconditions are intentionally strict to surface caller bugs early.

---

## 13. Complexity & Performance
- Per action: O(1) bookkeeping + O(N) next‑to‑act scan (`N=6`).  
- Showdown: O(#survivors) evaluator calls; Python fallback enumerates C(7,5)=21 per survivor.  
- For high‑throughput RL, prefer Rust evaluator and batch multiple envs at higher level.

---

## 14. Concurrency & Reentrancy
- `GameState` is mutable; treat as single‑threaded. If parallelizing, use *independent* engine instances per thread/process.  
- Do not retain external references to mutable lists inside `GameState` if you implement custom wrappers; copy if you must store snapshots.

---

## 15. Testing & QA (Recommended)
- **Property tests:**  
  - After any legal action: `pot == sum(cont)`; `current_bet == max(bet)`; fields non‑negative.  
  - On showdown: `sum(RL) == 0`.  
  - Carry‑down: removing a folded player’s hole cards must not change allocation amounts.  
- **Fuzzing:** random policies vs oracle pot allocator on synthetic hands.  
- **Seeded determinism:** given same seed+action trace, stacks and rewards must match byte‑for‑byte.

---

## 16. Versioning & Compatibility
- Semantic Versioning: **MAJOR.MINOR.PATCH**.  
- **Stable API:** `nlhe.core.*` and `nlhe.envs.*` are versioned. Adding fields to `LegalActionInfo` is allowed (backward‑compatible). Changing `actions_log` tuple shape is a **breaking** change.

---

## 17. Code Examples
### 17.1 Run a Hand (Headless)
```python
from nlhe.core.engine import NLHEngine
from nlhe.core.types import Action, ActionType

eng = NLHEngine(sb=1, bb=2, start_stack=100)
s = eng.reset_hand(button=0)
done = False; rewards = None
while not done:
    info = eng.legal_actions(s)
    # naïve policy: check/call else fold
    if any(a.kind==ActionType.CHECK for a in info.actions):
        a = Action(ActionType.CHECK)
    elif any(a.kind==ActionType.CALL for a in info.actions):
        a = Action(ActionType.CALL)
    else:
        a = Action(ActionType.FOLD)
    s, done, rewards, _ = eng.step(s, a)
    if s.next_to_act is None and not done:
        done, rewards = eng.advance_round_if_needed(s)
print("Final:", rewards)
```

### 17.2 Gym Env Loop
```python
import gymnasium as gym
from nlhe.envs.gym_env import NLHEGymEnv

env = NLHEGymEnv(hero_seat=0, seed=42)
obs, info = env.reset()
terminated = False
while not terminated:
    mask = obs["action_mask"]
    a = int(max((i for i,m in enumerate(mask) if m), default=1))  # pick any legal, fallback CHECK
    obs, r, terminated, truncated, info = env.step(a)
print("R:", r, "info:", info.get("rewards_all"))
```

### 17.3 Param Env Loop
```python
from nlhe.envs.param_env import NLHEParamEnv

env = NLHEParamEnv(hero_seat=0, seed=42)
obs, info = env.reset()
done = False
while not done:
    act = {"atype": 1, "r": 0.0}  # try CHECK
    obs, r, done, _, _ = env.step(act)
print("R:", r)
```

---

## 18. Extension Points
- **Evaluation backend:** replace `best5_rank_from_7` with higher‑performance FFI or vectorized kernels.  
- **Policies:** implement new `Agent` classes without touching engine.  
- **Envs:** create multi‑agent or parameterized discrete action envs by reusing `LegalActionInfo` and `advance_round_if_needed`.

---

## 19. Security & Validation
- All inputs are validated with assertions in debug contexts; for untrusted environments, prefer explicit exceptions and wrap calls with try/except and sanity checks.

---

## 20. Change Log (since 0.1.0)
- Initial public spec: extracted stable types, documented raise‑rights model, defined action log schema, and wrapper semantics.
