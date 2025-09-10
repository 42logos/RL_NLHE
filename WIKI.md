# NLHE Poker Engine - Comprehensive Wiki

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start Guide](#quick-start-guide)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [Agents System](#agents-system)
6. [Environment Wrappers](#environment-wrappers)
7. [Rust Extensions](#rust-extensions)
8. [Training & Reinforcement Learning](#training--reinforcement-learning)
9. [API Reference](#api-reference)
10. [Development Guide](#development-guide)
11. [Testing & Benchmarking](#testing--benchmarking)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)

## Introduction

The NLHE Poker Engine is a deterministic 6-max No-Limit Texas Hold'em engine designed for reinforcement learning research and production simulators. It features modular Python and Rust components, Gymnasium environments, and comprehensive training utilities.

### Key Features
- **Deterministic Gameplay**: Reproducible outcomes with fixed seeds
- **6-Max Support**: Full 6-player table implementation
- **Performance Optimized**: Rust extensions for critical paths
- **RL Ready**: Gymnasium compatibility for reinforcement learning
- **Modular Design**: Easy extension and customization
- **Comprehensive Testing**: Extensive test suite covering all game aspects

## Quick Start Guide

### Prerequisites
- Python 3.11+
- Rust toolchain (for Rust extensions)
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/42logos/RL_nlhe.git
cd RL_nlhe
```

2. **Install with pip** (builds Rust extensions automatically):
```bash
pip install -e .
```

3. **Alternative: Manual Rust build**:
```bash
python nlhe/build_rust.py --use-maturin --crate-dir nlhe_eval
python nlhe/build_rust.py --use-maturin --crate-dir rs_engine
```

### Verify Installation

```python
import nlhe
from nlhe.core.engine import NLHEngine

# Test basic functionality
engine = NLHEngine(sb=1, bb=2, start_stack=100)
state = engine.reset_hand()
print("Engine initialized successfully!")
```

## Architecture Overview

### High-Level Architecture
```
nlhe/
├── core/           # Core game engine and types
├── agents/         # Player agent implementations
├── envs/           # Gymnasium environment wrappers
├── demo/           # Demonstration scripts
├── nlhe_eval/      # Rust hand evaluator (PyO3)
├── rs_engine/      # Rust engine backend (PyO3)
└── train/          # Reinforcement learning utilities
```

### Data Flow
1. **Initialization**: Game state created with `reset_hand()`
2. **Action Processing**: `step()` method handles player actions
3. **Round Advancement**: `advance_round_if_needed()` manages game progression
4. **Showdown**: `_settle_showdown()` determines winners and distributes pots
5. **Reward Calculation**: Net winnings computed for each player

## Core Components

### Game State Management

#### GameState Dataclass
```python
@dataclass
class GameState:
    button: int                    # Dealer button position
    round_label: str               # Current round (Preflop, Flop, Turn, River)
    board: List[int]               # Community cards
    undealt: List[int]             # Remaining deck cards
    players: List[PlayerState]     # Player states
    current_bet: int               # Current highest bet
    min_raise: int                 # Minimum raise amount
    tau: int                       # Betting round identifier
    next_to_act: Optional[int]     # Next player to act
    step_idx: int                  # Action counter
    pot: int                       # Total pot size
    sb: int                        # Small blind amount
    bb: int                        # Big blind amount
    actions_log: List[Tuple]       # Action history
```

#### PlayerState Dataclass
```python
@dataclass
class PlayerState:
    hole: Optional[Tuple[int, int]]  # Hole cards
    stack: int                       # Remaining chips
    bet: int                         # Current round bet
    cont: int                        # Total hand contribution
    status: str                      # active, folded, or allin
    rho: int                         # Betting rights tracker
```

### Action System

#### Action Types
```python
class ActionType(Enum):
    FOLD = auto()     # Fold hand
    CHECK = auto()    # Check (no bet)
    CALL = auto()     # Call current bet
    RAISE_TO = auto() # Raise to specific amount
```

#### Legal Actions
The `legal_actions()` method returns a `LegalActionInfo` object containing:
- Available actions
- Minimum and maximum raise amounts
- Raise rights information

### Engine Workflow

1. **Initialization**:
```python
engine = NLHEngine(sb=1, bb=2, start_stack=100)
state = engine.reset_hand(button=0)
```

2. **Game Loop**:
```python
while not done:
    legal_info = engine.legal_actions(state)
    action = choose_action(legal_info)  # Your logic here
    state, done, rewards, info = engine.step(state, action)
```

3. **Round Advancement**:
```python
# Automatic round progression
done, rewards = engine.advance_round_if_needed(state)
```

## Agents System

### Agent Protocol
All agents must implement the `Agent` protocol:

```python
class Agent(Protocol):
    def act(self, env: EngineLike, state: GameState, seat: int) -> Action:
        """Return an action based on current game state"""
```

### Built-in Agents

#### TamedRandomAgent
Stochastic agent with configurable behavior:
- Raise probability control
- All-in probability control
- Bet sizing limits

```python
from nlhe.agents.tamed_random import TamedRandomAgent

agent = TamedRandomAgent(
    raise_prob=0.3,
    allin_prob=0.1,
    raise_cap=4  # Max raise multiple of pot
)
```

#### HumanCLIAgent
Command-line interface for manual play:
```python
from nlhe.agents.human_cli import HumanAgent

agent = HumanAgent()
```

### Creating Custom Agents

Example custom agent template:

```python
from nlhe.agents.base import Agent
from nlhe.core.types import Action, ActionType

class MyCustomAgent(Agent):
    def __init__(self, aggression: float = 0.5):
        self.aggression = aggression
        
    def act(self, env, state, seat):
        legal_info = env.legal_actions(state)
        # Your decision logic here
        if self.aggression > 0.7:
            return Action(ActionType.RAISE_TO, amount=state.pot)
        else:
            return Action(ActionType.CALL)
```

## Environment Wrappers

### Gymnasium Environment

#### NLHEGymEnv
Full Gymnasium compatibility for RL training:

```python
from nlhe.envs.gym_env import NLHEGymEnv

env = NLHEGymEnv(hero_seat=0, seed=42)
obs, info = env.reset()

# Standard Gym interface
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

**Observation Space**:
- Hero hole cards
- Community board
- Stack sizes
- Bet amounts
- Player statuses
- Action mask

**Action Space**: `Discrete(7)` representing:
- Fold, Check, Call
- Raise to min, min+2bb, min+4bb, all-in

### Parameterized Environment

#### NLHEParamEnv
Continuous action space for advanced RL:

```python
from nlhe.envs.param_env import NLHEParamEnv

env = NLHEParamEnv(hero_seat=0, seed=42)
obs, info = env.reset()

# Continuous action control
action = {
    "atype": 2,  # Action type index
    "r": 0.75    # Raise amount fraction (0.0-1.0)
}
obs, reward, done, info = env.step(action)
```

## Rust Extensions

### Performance Benefits
- **10-100x speedup** for hand evaluation
- **2-5x speedup** for engine operations
- **Reduced memory footprint**

### Hand Evaluator (nlhe_eval)

Rust implementation of 7-card hand evaluation:

```python
from nlhe.core.eval import best5_rank_from_7

# Automatically uses Rust extension if available
cards = [0, 1, 13, 14, 26, 39, 51]  # Example cards
rank = best5_rank_from_7(cards)
```

### Rust Engine (rs_engine)

Drop-in replacement for Python engine:

```python
from nlhe.core.rs_engine import NLHEngineRS

engine = NLHEngineRS(sb=1, bb=2, start_stack=100)
# Same API as Python engine
```

### Building Rust Components

```bash
# Build hand evaluator
cd nlhe/nlhe_eval
maturin develop --release

# Build Rust engine
cd nlhe/rs_engine  
maturin develop --release
```

## Training & Reinforcement Learning

### RLlib Integration

#### PPO Training Example

```python
from nlhe.train.PPOv2 import train_ppo

config = {
    "env": "NLHEGymEnv",
    "env_config": {
        "hero_seat": 0,
        "seed": 42
    },
    "framework": "torch",
    "num_workers": 4,
    "num_gpus": 1,
    "lr": 0.0001,
    "train_batch_size": 4000,
}

results = train_ppo(config)
```

### Training Utilities

#### Callbacks
```python
from nlhe.train.callbacks import NLHECallbacks

callbacks = NLHECallbacks(
    eval_freq=1000,
    checkpoint_freq=5000
)
```

#### Evaluators
```python
from nlhe.train.Evaluators import PolicyEvaluator

evaluator = PolicyEvaluator(
    num_episodes=100,
    opponent_types=["random", "tamed"]
)
```

### Hyperparameter Optimization

```python
from nlhe.train.PPOv2 import optimize_hyperparameters

study = optimize_hyperparameters(
    n_trials=100,
    direction="maximize"
)
```

## API Reference

### Core Engine (nlhe.core.engine)

#### NLHEngine Class
The main poker engine class that handles game state management, action processing, and round progression. The engine is deterministic given the same initial state and action sequence.

**Constructor**:
```python
NLHEngine(sb: int = 1, bb: int = 2, start_stack: int = 100, 
          num_players: int = 6, rng: Optional[random.Random] = None)
```
- **sb**: Small blind amount (default: 1), must be ≥1
- **bb**: Big blind amount (default: 2), must be ≥1  
- **start_stack**: Starting chip stack for all players (default: 100), must be positive
- **num_players**: Number of players at the table (fixed to 6)
- **rng**: Random number generator instance for reproducibility. If None, uses a new random.Random()

**Key Methods**:

##### `reset_hand(button: int = 0) -> GameState`
Initializes a new poker hand with shuffled deck, dealt hole cards, and posted blinds.
- **button**: Dealer button position (0-5)
- **Returns**: Complete initial GameState for the new hand
- **Side Effects**: Shuffles deck, deals hole cards, posts blinds, sets initial betting state
- **Invariants**: Ensures `pot == sb + bb` and `current_bet == bb` after reset

##### `owed(s: GameState, i: int) -> int`
Calculates the amount a player needs to call the current bet.
- **s**: Current game state
- **i**: Player seat index
- **Returns**: Chips needed to call (0 if player has already matched the current bet)
- **Formula**: `max(0, current_bet - players[i].bet)`

##### `legal_actions(s: GameState) -> LegalActionInfo`
Determines all legal actions available to the current player based on game rules and state.
- **s**: Current game state
- **Returns**: LegalActionInfo object with actions, raise bounds, and rights
- **Action Availability**:
  - `FOLD`: Available if player owes chips (`owed > 0`)
  - `CHECK`: Available if player owes nothing (`owed == 0`)
  - `CALL`: Available if player owes chips (`owed > 0`)
  - `RAISE_TO`: Available if player can raise (has chips and raise rights)
- **Raise Calculations**:
  - `min_raise_to`: Minimum raise amount (current_bet + min_raise if betting, else max(min_raise, 1))
  - `max_raise_to`: Maximum raise amount (player's bet + stack)
  - `has_raise_right`: True if player has raise rights (see Raise Rights Reopening below)

##### `step(s: GameState, a: Action) -> Tuple[GameState, bool, Optional[List[int]], Dict[str, Any]]`
Executes a single action and advances the game state.
- **s**: Current game state
- **a**: Action to execute (must be legal)
- **Returns**: Updated state, completion flag, rewards (if hand ended), and info dict
- **Validation**: Checks action legality based on game rules
- **State Updates**: Modifies player bets, stacks, status, and game state variables
- **Action Logging**: Appends action to `actions_log` with format `(seat, action_id, amount, round_id)`
- **Round Advancement**: Automatically advances rounds if betting completes

##### `advance_round_if_needed(s: GameState) -> Tuple[bool, Optional[List[int]]]`
Advances the betting round or triggers showdown when appropriate.
- **s**: Current game state
- **Returns**: Completion flag and player rewards if hand ended
- **Logic**: Checks if betting round is complete, deals next street or settles showdown
- **Showdown**: Resolves hand using carry-down side pot allocation

#### Raise Rights Reopening (ρ/τ Model)
The engine uses a raise rights system to manage when players can raise:
- **rho (ρ)**: Each player has a `rho` value representing the last step when they responded
- **tau (τ)**: The current round token indicating the last step when a full raise occurred
- **Raise Rights**: A player has raise rights if `rho < tau` (haven't responded since last full raise) or if `current_bet == 0` (opening bet)
- **Full Raise**: A raise that increases the bet by at least `min_raise` amount. This sets `tau = current_step` and resets other players' `rho` to -∞, reopening rights
- **Short All-in**: An all-in that doesn't meet the minimum raise amount doesn't reopen rights

#### Showdown & Side-Pot Allocation
The engine uses a carry-down algorithm for side pot allocation:
1. **Levels**: Identify unique contribution levels from all players
2. **Tranche Processing**: For each level, calculate the pot portion and determine eligible winners
3. **Carry-Down**: If no eligible winners at a level, chips carry down to next level
4. **Splitting**: Chips are split equally among winners, with remainder distributed clockwise from button
5. **Zero-Sum**: Final rewards ensure `sum(rewards) == 0` (net chip conservation)

**Invariants**:
- `pot == sum(player.cont)` at all times
- `current_bet == max(player.bet)` after each action
- `sum(rewards) == 0` after showdown

#### Error Handling
- **AssertionError**: Raised for invariant violations or incorrect API usage
- **ValueError**: Raised for malformed actions (e.g., missing amount for RAISE_TO)
- **Validation**: All inputs are validated; engine methods never raise on legal usage

#### Performance Characteristics
- **Time Complexity**: O(1) per action with O(N) scans for next player (N=6)
- **Showdown**: O(S) evaluator calls where S is number of survivors
- **Memory**: GameState contains all state; no external dependencies
- **Determinism**: Output determined by initial state and action sequence

#### Thread Safety
- **Not Thread-Safe**: GameState is mutable and should not be shared across threads
- **Recommendation**: Use separate engine instances for parallel processing

### Data Types (nlhe.core.types)

#### ActionType Enum
```python
class ActionType(Enum):
    FOLD = auto()    # Fold hand and forfeit pot
    CHECK = auto()   # Pass action without betting
    CALL = auto()    # Match current bet amount
    RAISE_TO = auto() # Increase bet to specified amount
```

#### Action Dataclass
```python
@dataclass
class Action:
    kind: ActionType       # Type of action
    amount: Optional[int] = None  # Amount for RAISE_TO actions
```

#### PlayerState Dataclass
```python
@dataclass
class PlayerState:
    hole: Optional[Tuple[int, int]] = None  # Hole cards
    stack: int = 0                          # Remaining chips
    bet: int = 0                            # Current round bet
    cont: int = 0                           # Total hand contribution
    status: str = "active"                  # active, folded, or allin
    rho: int = -10**9                       # Betting rights tracker
```

#### GameState Dataclass
```python
@dataclass
class GameState:
    button: int                             # Dealer button position
    round_label: str                        # Preflop, Flop, Turn, River
    board: List[int]                        # Community cards
    undealt: List[int]                      # Remaining deck cards
    players: List[PlayerState]              # All player states
    current_bet: int                        # Current highest bet
    min_raise: int                          # Minimum raise amount
    tau: int                                # Betting round identifier
    next_to_act: Optional[int]              # Next player to act
    step_idx: int                           # Action counter
    pot: int                                # Total pot size
    sb: int                                 # Small blind amount
    bb: int                                 # Big blind amount
    actions_log: List[Tuple[int, int, int, int]]  # Action history
```

#### LegalActionInfo Dataclass
```python
@dataclass
class LegalActionInfo:
    actions: List[Action]                   # Available legal actions
    min_raise_to: Optional[int] = None      # Minimum raise amount
    max_raise_to: Optional[int] = None      # Maximum raise amount
    has_raise_right: Optional[bool] = None  # Raise rights status
```

### Card Utilities (nlhe.core.cards)

#### `rank_of(c: int) -> int`
Returns the rank of a card (2-14, where 11=Jack, 12=Queen, 13=King, 14=Ace).
- **c**: Card integer (0-51)
- **Returns**: Card rank (2-14)

#### `suit_of(c: int) -> int` 
Returns the suit of a card (0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs).
- **c**: Card integer (0-51)
- **Returns**: Card suit (0-3)

#### `make_deck() -> List[int]`
Creates a standard 52-card deck as a list of integers 0-51.
- **Returns**: List of card integers in order

### Hand Evaluation (nlhe.core.eval)

#### `best5_rank_from_7(cards: List[int]) -> int`
Evaluates the best 5-card hand from 7 cards and returns a strength rank.
- **cards**: List of 7 card integers
- **Returns**: Hand strength rank (higher is better)

**Card Encoding**:
- Cards 0-12: 2♠, 3♠, 4♠, ..., A♠
- Cards 13-25: 2♥, 3♥, 4♥, ..., A♥  
- Cards 26-38: 2♦, 3♦, 4♦, ..., A♦
- Cards 39-51: 2♣, 3♣, 4♣, ..., A♣

### Agents System (nlhe.agents)

#### Agent Protocol
```python
class Agent(Protocol):
    def act(self, env: EngineLike, s: GameState, seat: int) -> Action:
        """
        Return an action based on current game state.
        
        Args:
            env: Engine-like object providing legal_actions and owed methods
            s: Current game state
            seat: Player's seat position
            
        Returns:
            Action: Chosen action to execute
        """
```

#### EngineLike Protocol
```python
class EngineLike(Protocol):
    def legal_actions(self, s: GameState) -> LegalActionInfo: ...
    def owed(self, s: GameState, i: int) -> int: ...
```

### Environment Wrappers

#### NLHEGymEnv (nlhe.envs.gym_env)
Gymnasium-compatible environment for reinforcement learning.

**Constructor**:
```python
NLHEGymEnv(hero_seat: int = 0, seed: Optional[int] = None,
           sb: int = 1, bb: int = 2, start_stack: int = 100,
           bot_kwargs: Optional[dict] = None)
```

**Observation Space**:
```python
spaces.Dict({
    "hero_hole": spaces.Box(low=-1, high=51, shape=(2,), dtype=np.int32),
    "board": spaces.Box(low=-1, high=51, shape=(5,), dtype=np.int32),
    "stacks": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int32),
    "bets": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int32),
    "conts": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int32),
    "status": spaces.MultiDiscrete([3]*6),
    "button": spaces.Discrete(6),
    "next_to_act": spaces.Discrete(6),
    "round": spaces.Discrete(5),
    "current_bet": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int32),
    "min_raise": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int32),
    "action_mask": spaces.MultiBinary(7),
})
```

**Action Space**: `spaces.Discrete(7)` with actions:
- 0: Fold
- 1: Check  
- 2: Call
- 3: Raise to minimum
- 4: Raise to minimum + 2*bb
- 5: Raise to minimum + 4*bb
- 6: All-in

#### NLHEParamEnv (nlhe.envs.param_env)
Parameterized environment with continuous action space.

**Constructor**:
```python
NLHEParamEnv(config: Optional[dict] = None, **kwargs)
```

**Observation Space**:
```python
spaces.Dict({
    "pot": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
    "current_bet": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
    "board": spaces.Box(low=-1, high=51, shape=(5,), dtype=np.int32),
    "board_len": spaces.Discrete(6),
    "hero_stack": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
    "hero_bet": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
    "hero_cont": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
    "hero_hole": spaces.Box(low=-1, high=51, shape=(2,), dtype=np.int32),
    "history": spaces.Box(low=-1, high=10_000, shape=(H, 4), dtype=np.int32),
})
```

**Action Space**:
```python
spaces.Dict({
    "atype": spaces.Discrete(4),  # Action type: 0=FOLD, 1=CHECK, 2=CALL, 3=RAISE
    "r": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),  # Raise fraction
})
```

### Rust Extensions

#### NLHEngineRS (nlhe.core.rs_engine)
Rust-accelerated drop-in replacement for NLHEngine with identical API.

```python
from nlhe.core.rs_engine import NLHEngineRS

engine = NLHEngineRS(sb=1, bb=2, start_stack=100)
# Same methods as NLHEngine: reset_hand, owed, legal_actions, step, advance_round_if_needed
```

#### Hand Evaluator (nlhe_eval)
Rust-optimized hand evaluation exposed via PyO3.

```python
from nlhe.core.eval import best5_rank_from_7  # Automatically uses Rust if available
```

### Training Utilities (nlhe.train)

#### PPOv2 Training
```python
from nlhe.train.PPOv2 import train_ppo

config = {
    "env": "NLHEGymEnv",
    "env_config": {"hero_seat": 0, "seed": 42},
    "framework": "torch",
    "num_workers": 4,
    "num_gpus": 1,
    "lr": 0.0001,
    "train_batch_size": 4000,
}

results = train_ppo(config)
```

#### Callbacks
```python
from nlhe.train.callbacks import NLHECallbacks

callbacks = NLHECallbacks(
    eval_freq=1000,        # Evaluation frequency
    checkpoint_freq=5000,  # Checkpoint frequency
)
```

#### Policy Evaluator
```python
from nlhe.train.Evaluators import PolicyEvaluator

evaluator = PolicyEvaluator(
    num_episodes=100,              # Evaluation episodes
    opponent_types=["random", "tamed"]  # Opponent strategies
)
```

### Advanced Usage Examples

#### Custom Agent Implementation
```python
from nlhe.agents.base import Agent
from nlhe.core.types import Action, ActionType

class ConservativeAgent(Agent):
    def __init__(self, call_threshold: float = 0.6):
        self.call_threshold = call_threshold
        
    def act(self, env, state, seat):
        legal_info = env.legal_actions(state)
        owe = env.owed(state, seat)
        
        # Only call if we have strong enough hand
        if owe > 0 and any(a.kind == ActionType.CALL for a in legal_info.actions):
            if random.random() < self.call_threshold:
                return Action(ActionType.CALL)
            else:
                return Action(ActionType.FOLD)
                
        # Check if possible
        if owe == 0 and any(a.kind == ActionType.CHECK for a in legal_info.actions):
            return Action(ActionType.CHECK)
            
        # Default to fold if no better option
        return Action(ActionType.FOLD)
```

#### Environment Integration
```python
from nlhe.envs.gym_env import NLHEGymEnv
from stable_baselines3 import PPO

env = NLHEGymEnv(hero_seat=0, seed=42)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("nlhe_ppo_model")
```

#### State Analysis Utility
```python
def analyze_game_state(state: GameState):
    """Comprehensive state analysis for debugging and AI training."""
    analysis = {
        "round": state.round_label,
        "pot_size": state.pot,
        "current_bet": state.current_bet,
        "active_players": sum(1 for p in state.players if p.status == 'active'),
        "player_stacks": [p.stack for p in state.players],
        "player_contributions": [p.cont for p in state.players],
        "board_cards": len(state.board),
        "remaining_deck": len(state.undealt),
    }
    
    if state.next_to_act is not None:
        player = state.players[state.next_to_act]
        analysis["next_player"] = {
            "seat": state.next_to_act,
            "stack": player.stack,
            "owed": state.current_bet - player.bet,
            "hole_cards": player.hole
        }
    
    return analysis
```

## Development Guide

### Project Structure
```
nlhe/
├── core/           # Core game logic
│   ├── engine.py   # Main game engine
│   ├── types.py    # Data classes
│   ├── cards.py    # Card utilities
│   ├── eval.py     # Hand evaluation
│   └── rs_engine.py # Rust engine wrapper
├── agents/         # Player agents
├── envs/           # Environment wrappers
├── demo/           # Demonstration scripts
├── train/          # Training utilities
└── rs_engine/      # Rust engine implementation
```

### Adding New Features

#### 1. Custom Agents
Create new agent classes in `nlhe/agents/` implementing the `Agent` protocol.

#### 2. Environment Variants
Extend base environments in `nlhe/envs/` for custom observation/action spaces.

#### 3. Training Algorithms
Add new training scripts in `nlhe/train/` following existing patterns.

### Code Style Guidelines

- **Type Hints**: Always use type annotations
- **Docstrings**: Google-style docstrings for all public methods
- **Testing**: Write tests for new features in `tests/`
- **Performance**: Use Rust extensions for performance-critical code

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug engine for step-by-step execution
from nlhe.core.engine import NLHEngine
engine = NLHEngine()
state = engine.reset_hand()

# Add breakpoints in critical methods
breakpoint()  # Python 3.7+
```

## Testing & Benchmarking

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_engine_step.py -v

# Run with coverage
pytest --cov=nlhe tests/
```

### Test Categories

- **Engine Tests**: Core game logic validation
- **Parity Tests**: Python vs Rust engine consistency
- **Performance Tests**: Speed comparisons
- **Integration Tests**: Full game scenarios

### Benchmarking

```bash
# Run performance benchmarks
python tests/speed/compare_engines.py

# Profile specific components
python tests/speed/profile_components.py
```

### Performance Metrics

| Component | Python (ops/sec) | Rust (ops/sec) | Speedup |
|-----------|------------------|----------------|---------|
| Hand Eval | 50,000           | 5,000,000      | 100x    |
| Step      | 10,000           | 50,000         | 5x      |
| Reset     | 1,000            | 5,000          | 5x      |

## Troubleshooting

### Common Issues

#### 1. Rust Build Failures
**Problem**: `maturin` fails to build Rust extensions
**Solution**: Ensure Rust toolchain is installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### 2. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'nlhe'`
**Solution**: Install in development mode:
```bash
pip install -e .
```

#### 3. Performance Issues
**Problem**: Slow hand evaluation
**Solution**: Ensure Rust extensions are built:
```bash
python nlhe/build_rust.py --crate-dir nlhe_eval
```

#### 4. Gymnasium Compatibility
**Problem**: Environment registration issues
**Solution**: Check Gymnasium version:
```bash
pip install gymnasium>=0.29
```

### Debugging Game State

```python
# Print game state for debugging
def print_state(state):
    print(f"Round: {state.round_label}")
    print(f"Pot: {state.pot}")
    print(f"Board: {state.board}")
    for i, player in enumerate(state.players):
        print(f"Player {i}: {player.status}, stack: {player.stack}, bet: {player.bet}")
```

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**:
```bash
git checkout -b feature/your-feature
```
3. **Implement changes** with tests
4. **Run tests and benchmarks**:
```bash
pytest tests/
python tests/speed/compare_engines.py
```
5. **Submit pull request**

### Code Review Guidelines

- **Tests required** for all new features
- **Performance impact** assessment for changes
- **Documentation updates** for new functionality
- **Backward compatibility** maintained

### Release Process

1. **Version bump** in `pyproject.toml`
2. **Update changelog** in `README.md`
3. **Run full test suite**
4. **Build and test Rust extensions**
5. **Create release tag**
6. **Publish to PyPI** (if applicable)

### Community Guidelines

- **Be respectful** in discussions
- **Provide reproducible examples** for bug reports
- **Document your contributions** thoroughly
- **Help others** learn and contribute

## Additional Resources

### Documentation
- [API Specification](API_SPEC.md)
- [MkDocs Site](docs/)
- [Example Notebooks](examples/)

### Related Projects
- [OpenSpiel](https://github.com/deepmind/open_spiel) - Google's game framework
- [PokerRL](https://github.com/TinkeringCode/PokerRL) - Poker reinforcement learning
- [PyPokerEngine](https://github.com/ishikota/PyPokerEngine) - Python poker engine

### Learning Materials
- [No-Limit Hold'em Rules](https://en.wikipedia.org/wiki/Texas_hold_%27em)
- [Reinforcement Learning Basics](https://spinningup.openai.com/)
- [Rust for Python Developers](https://github.com/rochacbruno/rust-python)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- [GitHub Issues](https://github.com/42logos/RL_nlhe/issues)
- [Discussion Forum](https://github.com/42logos/RL_nlhe/discussions)
- [Email](mailto:maintainer@example.com)

---

*This wiki is maintained by the NLHE Poker Engine community. Last updated: 2025-09-10*
