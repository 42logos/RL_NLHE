# NLHE 6-Max Poker Engine: A Platform for Reinforcement Learning in Imperfect Information Games

**Version:** 0.1.0 | **License:** MIT | **Status:** Beta

## Abstract

This project presents a high-performance, deterministic 6-max No-Limit Texas Hold'em poker engine designed specifically for reinforcement learning research. The platform features modular Python and Rust components, Gymnasium-compatible environments, and comprehensive training utilities. The engine enforces complete poker rules, generates legal actions, progresses through betting rounds, and resolves showdowns with side-pot support, providing a robust foundation for studying imperfect information games, multi-agent systems, and decision-making under uncertainty.

## 1. Introduction

No-Limit Texas Hold'em represents one of the most challenging domains in artificial intelligence research due to its combination of imperfect information, stochastic outcomes, and complex strategic interactions. This project addresses the need for a research-grade poker engine that balances computational efficiency with academic rigor.

### 1.1 Research Objectives

- Provide a mathematically precise implementation of NLHE game mechanics
- Enable reproducible research in reinforcement learning for imperfect information games
- Offer multiple environment interfaces suitable for different research paradigms
- Support both classical and deep reinforcement learning approaches
- Facilitate performance benchmarking and algorithmic comparisons

### 1.2 Mathematical Foundations

The poker game can be formalized as an extensive-form game with imperfect information. Let:

- $\mathcal{P} = \{1, 2, \dots, 6\}$ be the set of players
- $\mathcal{S}$ be the state space encompassing all possible game configurations
- $\mathcal{A}(s)$ be the set of legal actions available in state $s \in \mathcal{S}$
- $\mathcal{I}_i$ be the information set for player $i$, representing states indistinguishable to $i$
- $u_i(\sigma)$ be the expected utility for player $i$ under strategy profile $\sigma$

The engine implements the complete game tree traversal with efficient state representation and action validation.

## 2. System Architecture

### 2.1 Core Mathematical Model

The game state $s \in \mathcal{S}$ is defined by the tuple:

$$
s = (b, r, B, D, P, c_b, m_r, \tau, n_a, \iota, p, s_b, b_b, L)
$$

Where:
- $b$: Button position (dealer)
- $r$: Current round label $\in \{\text{Preflop}, \text{Flop}, \text{Turn}, \text{River}\}$
- $B$: Community cards on board
- $D$: Undealt cards remaining
- $P$: Player states $p_i = (h_i, s_i, b_i, c_i, \sigma_i, \rho_i)$ for each player $i$
- $c_b$: Current bet amount
- $m_r$: Minimum raise amount
- $\tau$: Raise threshold parameter
- $n_a$: Next player to act
- $\iota$: Step index
- $p$: Total pot size
- $s_b$: Small blind amount
- $b_b$: Big blind amount
- $L$: Action history log

### 2.2 Repository Structure

```
nlhe/
  core/           # Mathematical engine implementation, datatypes, card utilities
  agents/         # Agent protocols and baseline implementations
  envs/           # Gymnasium and parameterized environments
  demo/           # Demonstration and visualization utilities
  nlhe_eval/      # High-performance Rust hand evaluator (PyO3)
  rs_engine/      # Optimized Rust state machine backend (PyO3)
  train/          # Reinforcement learning training scripts
docs/             # Research documentation and API specifications
API_SPEC.md       # Formal mathematical API specification
```

## 3. Core Engine Implementation

### 3.1 Card Representation and Mathematics

Cards are represented as integers $c \in [0, 51]$ with mathematical mappings:

$$
\text{rank}(c) = \left\lfloor \frac{c}{13} \right\rfloor + 2 \quad \text{for } c \in [0, 51]
$$
$$
\text{suit}(c) = c \mod 13 \quad \text{for } c \in [0, 51]
$$

The deck generation follows a deterministic permutation:
$$
D = \text{shuffle}(\text{range}(0, 51), \text{seed})
$$

### 3.2 Formal Action Space Definition

The action space $\mathcal{A}$ is defined by the ActionType enumeration:

- $\text{FOLD}$: Player forfeits hand, $u_i = -c_i$
- $\text{CHECK}$: Pass action, requires $b_i = c_b$
- $\text{CALL}$: Match current bet, amount = $\max(0, c_b - b_i)$
- $\text{RAISE\_TO}$: Increase bet to amount $a$, with constraints:
  $$
  a > c_b \quad \text{and} \quad a \leq b_i + s_i
  $$
  $$
  a \geq c_b + m_r \quad \text{unless} \quad a = b_i + s_i \text{ (all-in)}
  $$

### 3.3 Hand Evaluation Theory

Hand strength is computed using the standard poker hand ranking system. For any 7-card combination $C$ (2 hole cards + 5 community cards), the engine computes:

$$
\text{rank}(C) = \max_{H \subseteq C, |H| = 5} \text{value}(H)
$$

Where $\text{value}(H)$ maps 5-card hands to an ordinal ranking following standard poker rules (Royal Flush > Straight Flush > ... > High Card).

### 3.4 Showdown Mathematics

The pot distribution algorithm implements the side-pot mechanism mathematically:

Let $L = \{l_1, l_2, \dots, l_k\}$ be the sorted contribution levels
For each level $l_j$:
- $P_j = |\{i: c_i \geq l_j\}| \cdot (l_j - l_{j-1}) + \text{carry}$
- Eligible players: $E_j = \{i: \sigma_i \neq \text{folded} \land c_i \geq l_j\}$
- Winners: $W_j = \{i \in E_j: \text{rank}_i = \max_{k \in E_j} \text{rank}_k\}$
- Distribution: $r_i \leftarrow r_i + \left\lfloor \frac{P_j}{|W_j|} \right\rfloor + \delta(i \in \text{first } (P_j \mod |W_j|) \text{ players by position})$

Ensuring the zero-sum property: $\sum_{i=1}^6 r_i = 0$

## 4. Reinforcement Learning Environments

### 4.1 Formal MDP Formulation

The poker environment can be modeled as a Partially Observable Markov Decision Process (POMDP):

- **State space**: $\mathcal{S}$ (fully observable to environment)
- **Observation space**: $\mathcal{O} \subset \mathcal{S}$ (partial information for agents)
- **Action space**: $\mathcal{A}$ (legal poker actions)
- **Transition function**: $T(s, a, s') = \mathbb{P}(s' | s, a)$
- **Reward function**: $R(s, a, s') =$ net chip change

### 4.2 NLHEGymEnv Observation Space

The observation for player $i$ includes:
- Hero's hole cards: $h_i$
- Community board: $B$
- Stack information: $\mathbf{s} = (s_1, s_2, \dots, s_6)$
- Bet information: $\mathbf{b} = (b_1, b_2, \dots, b_6)$
- Contribution information: $\mathbf{c} = (c_1, c_2, \dots, c_6)$
- Status information: $\boldsymbol{\sigma} = (\sigma_1, \sigma_2, \dots, \sigma_6)$
- Game context: $(b, n_a, r, c_b, m_r)$
- Action mask: $\mathbf{m} \in \{0,1\}^4$ indicating legal actions

### 4.3 NLHEParamEnv Continuous Formulation

This environment provides a continuous action representation suitable for policy gradient methods:

$$
\mathbf{a} = (\text{atype}, r) \quad \text{where } \text{atype} \in \{0,1,2,3\}, r \in [0,1]
$$

The raise amount is computed as:
$$
a_{\text{raise}} = \text{min\_raise} + r \cdot (\text{max\_raise} - \text{min\_raise})
$$

## 5. Research Applications and Contributions

### 5.1 Key Research Features

1. **Deterministic Gameplay**: Fixed random seeds enable perfect reproducibility
2. **Complete Information Logging**: Full action history for analysis and reconstruction
3. **Multiple Agent Interfaces**: Support for various RL paradigms and algorithms
4. **Performance Optimization**: Rust extensions for computational efficiency
5. **Mathematical Precision**: Formally verified game mechanics implementation

### 5.2 Experimental Validation

The engine has been validated through:
- Unit testing of all game mechanics
- Parity testing between Python and Rust implementations
- Performance benchmarking against established poker systems
- Reinforcement learning baseline experiments

### 5.3 Research Directions Enabled

1. **Multi-agent Reinforcement Learning**: Studying emergence of poker strategies
2. **Imperfect Information Game Theory**: Testing equilibrium concepts
3. **Meta-learning**: Adaptation to opponent strategies
4. **Explainable AI**: Interpreting learned poker policies
5. **Transfer Learning**: Applying policies across different game configurations

## 6. Usage Examples

### 6.1 Mathematical Engine Interface

```python
from nlhe.core.engine import NLHEngine
from nlhe.core.types import Action, ActionType
import numpy as np

# Initialize engine with mathematical parameters
eng = NLHEngine(sb=1, bb=2, start_stack=100, num_players=6)
state = eng.reset_hand(button=0, seed=42)  # Fixed seed for reproducibility

# Game theoretic value iteration
while not terminated:
    legal_info = eng.legal_actions(state)
    # Implement Nash equilibrium strategy or RL policy
    action = compute_optimal_action(state, legal_info)
    state, terminated, rewards, _ = eng.step(state, action)
    
    if state.next_to_act is None and not terminated:
        terminated, rewards = eng.advance_round_if_needed(state)

print("Expected value analysis:", np.array(rewards))
```

### 6.2 Reinforcement Learning Training

```python
from nlhe.envs.gym_env import NLHEGymEnv
from stable_baselines3 import PPO

# Create research environment
env = NLHEGymEnv(hero_seat=0, seed=42)

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=64)
model.learn(total_timesteps=1_000_000)

# Evaluate policy performance
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

## 7. Installation and Dependencies

### 7.1 Research Software Stack

- **Python 3.11+**: Primary implementation language
- **Rust toolchain**: For high-performance extensions
- **Gymnasium**: Standard RL environment interface
- **Ray RLlib**: Scalable distributed reinforcement learning
- **PyTorch**: Deep learning framework
- **Hydra**: Configuration management for experiments

### 7.2 Installation for Research Use

```bash
# Install with Rust extensions for maximum performance
pip install -e .[dev]  # Includes development dependencies

# Or install core functionality only
pip install -e .

# Build Rust components individually for research benchmarking
python nlhe/build_rust.py --use-maturin --crate-dir nlhe_eval
python nlhe/build_rust.py --use-maturin --crate-dir rs_engine
```

## 8. Experimental Methodology

### 8.1 Reproducibility Guidelines

All experiments should:
- Use fixed random seeds for deterministic behavior
- Report complete environment configuration parameters
- Include statistical significance measures
- Provide baseline comparisons against established algorithms
- Document computational resources and training time

### 8.2 Performance Metrics

Recommended evaluation metrics:
- **Expected Value**: Mean reward per hand
- **Win Rate**: Percentage of winning hands
- **Strategy Entropy**: Diversity of action choices
- **Exploration Efficiency**: Learning curve characteristics
- **Computational Efficiency**: Steps per second during training

## 9. Citation and Acknowledgments

If you use this platform in your research, please cite:

```bibtex
@software{nlhe_engine_2024,
  title = {NLHE 6-Max Poker Engine: A Research Platform for Reinforcement Learning},
  author = {Ylogos},
  year = {2025},
  url = {https://github.com/42logos/RL_nlhe},
  version = {0.1.0},
  license = {MIT}
}
```

## 10. Future Research Directions

- Integration with larger game theory frameworks
- Support for additional poker variants
- Advanced opponent modeling capabilities
- Real-time strategy adaptation
- Multi-modal observation spaces (including betting patterns)

## 11. License

MIT License - See LICENSE file for details.

This research software is provided for academic use only. Commercial use may require additional licensing.
