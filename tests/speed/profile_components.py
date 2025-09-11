#!/usr/bin/env python3
"""
Component-Level Performance Profiler for Rust Engine
===================================================

This script profiles individual components of the Rust engine to identify
performance bottlenecks and FFI overhead sources. It times each operation
at a granular level to understand where the slowdowns occur.
"""

import time
import random
import statistics
from typing import List, Dict, Any, Optional, Tuple
import sys
from contextlib import contextmanager

# Try to import both engines
PythonEngine = None
RustEngine = None
_rs = None

try:
    from nlhe.core.engine import NLHEngine as PythonEngine
    python_available = True
except ImportError:
    python_available = False

try:
    from nlhe.core.rs_engine import NLHEngine as RustEngine
    import nlhe_engine as _rs
    rust_available = True
except ImportError:
    rust_available = False

from nlhe.core.types import Action, ActionType


@contextmanager
def timer(name: str, results: Dict[str, List[float]]):
    """Context manager to time code blocks and store results"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    if name not in results:
        results[name] = []
    results[name].append(elapsed_ms)


class ComponentProfiler:
    """Profile individual components of the poker engines"""
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.rng = random.Random(42)  # Fixed seed for reproducibility
        
    def profile_python_engine(self) -> Dict[str, List[float]]:
        """Profile Python engine components"""
        if not python_available or PythonEngine is None:
            return {}
            
        print("Profiling Python Engine Components...")
        results = {}
        engine = PythonEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        
        # Profile reset_hand components
        for i in range(self.num_samples):
            with timer("python_reset_total", results):
                state = engine.reset_hand(i % 6)
        
        # Profile step components
        for i in range(self.num_samples):
            state = engine.reset_hand(i % 6)
            if state.next_to_act is not None:
                # Time legal_actions
                with timer("python_legal_actions", results):
                    legal = engine.legal_actions(state)
                
                if legal.actions:
                    # Select a safe action
                    valid_actions = []
                    for act in legal.actions:
                        if act.kind in [ActionType.FOLD, ActionType.CHECK, ActionType.CALL]:
                            valid_actions.append(act)
                        elif act.kind == ActionType.RAISE_TO and legal.min_raise_to is not None:
                            valid_actions.append(Action(ActionType.RAISE_TO, legal.min_raise_to))
                    
                    if valid_actions:
                        action = self.rng.choice(valid_actions)
                        
                        # Time step operation
                        with timer("python_step_total", results):
                            try:
                                state, done, rewards, info = engine.step(state, action)
                            except (ValueError, AssertionError):
                                continue  # Skip invalid actions
        
        return results
    
    def profile_rust_engine_detailed(self) -> Dict[str, List[float]]:
        """Profile Rust engine with detailed component breakdown"""
        if not rust_available or RustEngine is None or _rs is None:
            return {}
            
        print("Profiling Rust Engine Components (Detailed)...")
        results = {}
        engine = RustEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        
        # Profile reset_hand components
        for i in range(self.num_samples):
            with timer("rust_reset_total", results):
                state = engine.reset_hand(i % 6)
        
        # Profile individual reset components by accessing Rust directly
        rs_engine = _rs.NLHEngine(sb=1, bb=2, start_stack=200, num_players=6, seed=42)
        
        for i in range(self.num_samples):
            # Time the actual Rust reset call
            with timer("rust_reset_internal", results):
                s_rs = rs_engine.reset_hand(i % 6)
            
            # Time the Python object conversion
            with timer("rust_reset_conversion", results):
                from nlhe.core.types import PlayerState as PyPlayerState
                from nlhe.core.types import GameState as PyGameState
                
                players = []
                for pr in s_rs.players:
                    players.append(PyPlayerState(
                        hole=None if pr.hole is None else (int(pr.hole[0]), int(pr.hole[1])),
                        stack=int(pr.stack), bet=int(pr.bet), cont=int(pr.cont),
                        status=str(pr.status), rho=int(pr.rho)
                    ))
                s = PyGameState(
                    button=int(s_rs.button),
                    round_label=str(s_rs.round_label),
                    board=[int(x) for x in s_rs.board],
                    undealt=[int(x) for x in s_rs.undealt],
                    players=players,
                    current_bet=int(s_rs.current_bet),
                    min_raise=int(s_rs.min_raise),
                    tau=int(s_rs.tau),
                    next_to_act=None if s_rs.next_to_act is None else int(s_rs.next_to_act),
                    step_idx=int(s_rs.step_idx),
                    pot=int(s_rs.pot),
                    sb=1, bb=2,
                    actions_log=[(int(i),int(a),int(v),int(r)) for (i,a,v,r) in s_rs.actions_log],
                )
        
        # Profile step components
        for i in range(self.num_samples):
            state = engine.reset_hand(i % 6)
            if state.next_to_act is not None:
                # Time legal_actions
                with timer("rust_legal_actions_total", results):
                    legal = engine.legal_actions(state)
                
                # Time the Rust legal_actions call directly
                with timer("rust_legal_actions_internal", results):
                    legal_rs = rs_engine.legal_actions_now()
                
                # Time the conversion
                with timer("rust_legal_actions_conversion", results):
                    from nlhe.core.rs_engine import _from_rs_legal
                    legal_converted = _from_rs_legal(legal_rs)
                
                if legal.actions:
                    # Select a safe action (avoid raises that might be invalid)
                    valid_actions = []
                    for act in legal.actions:
                        if act.kind in [ActionType.FOLD, ActionType.CHECK, ActionType.CALL]:
                            valid_actions.append(act)
                        elif act.kind == ActionType.RAISE_TO and legal.min_raise_to is not None:
                            # Use minimum valid raise
                            valid_actions.append(Action(ActionType.RAISE_TO, legal.min_raise_to))
                    
                    if valid_actions:
                        action = self.rng.choice(valid_actions)
                        
                        # Time step operation total
                        with timer("rust_step_total", results):
                            try:
                                state, done, rewards, info = engine.step(state, action)
                            except ValueError:
                                continue  # Skip invalid actions
                        
                        # Time step components separately on a fresh state
                        fresh_state = engine.reset_hand(i % 6)
                        if fresh_state.next_to_act is not None:
                            fresh_legal = engine.legal_actions(fresh_state)
                            if fresh_legal.actions:
                                # Use same safe action selection
                                fresh_valid_actions = []
                                for act in fresh_legal.actions:
                                    if act.kind in [ActionType.FOLD, ActionType.CHECK, ActionType.CALL]:
                                        fresh_valid_actions.append(act)
                                    elif act.kind == ActionType.RAISE_TO and fresh_legal.min_raise_to is not None:
                                        fresh_valid_actions.append(Action(ActionType.RAISE_TO, fresh_legal.min_raise_to))
                                
                                if fresh_valid_actions:
                                    safe_action = fresh_valid_actions[0]  # Use first valid action for consistency
                                    
                                    # Time action conversion
                                    with timer("rust_step_action_conversion", results):
                                        from nlhe.core.rs_engine import _to_rs_action
                                        rs_action = _to_rs_action(safe_action)
                                    
                                    # Time the Rust step call
                                    with timer("rust_step_internal", results):
                                        try:
                                            done_rs, rewards_rs, diff_rs = rs_engine.step_diff(rs_action)
                                        except ValueError:
                                            continue  # Skip if still invalid
                                    
                                    # Time applying the diff
                                    with timer("rust_step_diff_apply", results):
                                        from nlhe.core.rs_engine import _apply_diff_inplace
                                        # Create a fresh state to apply diff to
                                        temp_state = engine.reset_hand(i % 6)
                                        _apply_diff_inplace(temp_state, diff_rs)
        
        return results
    
    def profile_ffi_overhead(self) -> Dict[str, List[float]]:
        """Profile pure FFI overhead without actual computation"""
        if not rust_available or _rs is None:
            return {}
            
        print("Profiling Pure FFI Overhead...")
        results = {}
        
        # Test simple data conversions
        for i in range(self.num_samples * 10):  # More samples for micro-operations
            # Time creating a simple Rust Action
            action = Action(ActionType.FOLD)
            with timer("ffi_action_creation", results):
                from nlhe.core.rs_engine import _to_rs_action
                rs_action = _to_rs_action(action)
            
            # Time accessing properties
            with timer("ffi_property_access", results):
                _ = rs_action.kind
                _ = rs_action.amount
        
        # Test object creation overhead
        rs_engine = _rs.NLHEngine(sb=1, bb=2, start_stack=200, num_players=6, seed=42)
        for i in range(self.num_samples):
            with timer("ffi_engine_creation", results):
                temp_engine = _rs.NLHEngine(sb=1, bb=2, start_stack=200, num_players=6, seed=42)
        
        return results
    
    def profile_memory_operations(self) -> Dict[str, List[float]]:
        """Profile memory-related operations"""
        print("Profiling Memory Operations...")
        results = {}
        
        if python_available:
            # Profile Python object creation
            for i in range(self.num_samples * 10):
                with timer("python_object_creation", results):
                    from nlhe.core.types import PlayerState, GameState
                    players = [PlayerState(stack=200) for _ in range(6)]
                    state = GameState(
                        button=0, round_label="Preflop", board=[], undealt=list(range(52)),
                        players=players, current_bet=2, min_raise=2, tau=0,
                        next_to_act=0, step_idx=0, pot=3, sb=1, bb=2
                    )
        
        if rust_available and _rs is not None:
            # Profile Rust object creation
            for i in range(self.num_samples * 10):
                with timer("rust_object_creation", results):
                    players_rs = []
                    for j in range(6):
                        players_rs.append(_rs.PlayerState(
                            hole=None, stack=200, bet=0, cont=0, status="active", rho=-1000000000
                        ))
                    state_rs = _rs.GameState(
                        button=0, round_label="Preflop", board=[], undealt=list(range(52)),
                        players=players_rs, current_bet=2, min_raise=2, tau=0,
                        next_to_act=0, step_idx=0, pot=3, sb=1, bb=2
                    )
        
        return results
    
    def profile_showdown_sidepot_detailed(self) -> Dict[str, List[float]]:
        """Profile showdown and sidepot calculation performance in detail"""
        results = {}
        
        if python_available and PythonEngine is not None:
            print("Profiling Python Showdown & Sidepot Components...")
            self._profile_python_showdown_components(results)
        
        if rust_available and RustEngine is not None and _rs is not None:
            print("Profiling Rust Showdown & Sidepot Components...")
            self._profile_rust_showdown_components(results)
        
        return results
    
    def _profile_python_showdown_components(self, results: Dict[str, List[float]]):
        """Profile Python showdown components in isolation"""
        if not python_available or PythonEngine is None:
            return
            
        engine = PythonEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        
        # Create various showdown scenarios
        test_scenarios = self._generate_showdown_scenarios(engine, is_rust=False)
        
        for scenario_name, states in test_scenarios.items():
            for state in states:
                if state is None:
                    continue
                    
                # Time the complete showdown process
                with timer(f"python_showdown_{scenario_name}", results):
                    try:
                        rewards = engine._settle_showdown(state)
                    except Exception:
                        continue
                
                # Profile individual showdown components
                if any(p.status != 'folded' for p in state.players):
                    # Time hand strength evaluation
                    with timer(f"python_hand_eval_{scenario_name}", results):
                        from nlhe.core.eval import best5_rank_from_7
                        for i, p in enumerate(state.players):
                            if p.status != 'folded' and p.hole is not None:
                                seven = list(p.hole) + state.board
                                best5_rank_from_7(seven)
                    
                    # Time sidepot calculation
                    with timer(f"python_sidepot_{scenario_name}", results):
                        # Extract the sidepot logic
                        levels = sorted({p.cont for p in state.players if p.cont > 0})
                        if levels:
                            y_prev = 0
                            for y in levels:
                                contributors_count = sum(1 for p in state.players if p.cont >= y)
                                Pk = contributors_count * (y - y_prev)
                                y_prev = y
                    
                    # Time winner determination
                    with timer(f"python_winner_determination_{scenario_name}", results):
                        A = [i for i, p in enumerate(state.players) if p.status != 'folded']
                        if A and len(state.board) == 5:  # Only for complete boards
                            from nlhe.core.eval import best5_rank_from_7
                            ranks = {}
                            for i in A:
                                hole = state.players[i].hole
                                if hole is not None:
                                    seven = list(hole) + state.board
                                    ranks[i] = best5_rank_from_7(seven)
                            if ranks:
                                best_val = max(ranks.values())
                                winners = [i for i in A if ranks.get(i) == best_val]
    
    def _profile_rust_showdown_components(self, results: Dict[str, List[float]]):
        """Profile Rust showdown components in isolation"""
        if not (rust_available and RustEngine is not None and _rs is not None):
            return
            
        engine = RustEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        rs_engine = _rs.NLHEngine(sb=1, bb=2, start_stack=200, num_players=6, seed=42)
        
        # Create various showdown scenarios
        test_scenarios = self._generate_showdown_scenarios(engine, is_rust=True)
        
        for scenario_name, states in test_scenarios.items():
            for state in states:
                if state is None:
                    continue
                    
                # Time the complete showdown process via wrapper
                with timer(f"rust_showdown_total_{scenario_name}", results):
                    try:
                        state_copy = self._deep_copy_state(state)
                        done, rewards = engine.advance_round_if_needed(state_copy)
                    except Exception:
                        continue
                
                # Time direct Rust showdown call
                if any(p.status != 'folded' for p in state.players):
                    try:
                        # Convert to Rust state for direct timing
                        rs_state = self._convert_to_rust_state(state, rs_engine)
                        
                        with timer(f"rust_showdown_internal_{scenario_name}", results):
                            # Force showdown by setting next_to_act to None and calling advance
                            rs_state.next_to_act = None
                            done_rs, rewards_rs = rs_engine.advance_round_if_needed_now()
                            
                    except Exception:
                        continue
    
    def _generate_showdown_scenarios(self, engine, is_rust: bool = False) -> Dict[str, List]:
        """Generate various showdown scenarios for testing"""
        scenarios = {
            'heads_up': [],
            'three_way': [],
            'sidepot_simple': [],
            'sidepot_complex': [],
            'allin_multiway': []
        }
        
        # Generate heads-up scenarios
        for _ in range(min(self.num_samples, 20)):
            try:
                state = engine.reset_hand(0)
                done = False  # Initialize done variable
                
                # Simulate heads-up to river
                for _ in range(15):  # Max actions to reach showdown
                    if state.next_to_act is None:
                        break
                    legal = engine.legal_actions(state)
                    if not legal.actions:
                        break
                    action = self._get_conservative_action(legal)
                    if action is None:
                        break
                    state, done, rewards, _ = engine.step(state, action)
                    if done:
                        break
                
                # Ensure we have a showdown scenario
                if (not done and state.next_to_act is None and 
                    sum(1 for p in state.players if p.status != 'folded') >= 2):
                    scenarios['heads_up'].append(state)
            except Exception:
                continue
        
        # Generate sidepot scenarios by manipulating states
        for _ in range(min(self.num_samples, 10)):
            try:
                state = engine.reset_hand(0)
                # Create artificial sidepot scenario
                state.players[0].cont = 50
                state.players[1].cont = 100
                state.players[2].cont = 150
                state.players[3].status = 'folded'
                state.players[4].status = 'folded'
                state.players[5].status = 'folded'
                state.next_to_act = None
                state.round_label = 'River'
                # Add full board
                if len(state.board) < 5:
                    while len(state.board) < 5 and state.undealt:
                        card = state.undealt.pop(0)
                        state.board.append(card)
                scenarios['sidepot_simple'].append(state)
            except Exception:
                continue
        
        return scenarios
    
    def _get_conservative_action(self, legal):
        """Get a conservative action that's likely to be valid"""
        for action in legal.actions:
            if action.kind == ActionType.CHECK:
                return action
            elif action.kind == ActionType.CALL:
                return action
        # If no check/call, try fold
        for action in legal.actions:
            if action.kind == ActionType.FOLD:
                return action
        # Last resort
        return legal.actions[0] if legal.actions else None
    
    def _deep_copy_state(self, state):
        """Create a deep copy of the game state"""
        import copy
        return copy.deepcopy(state)
    
    def _convert_to_rust_state(self, py_state, rs_engine):
        """Convert Python state to Rust state format"""
        # This is a simplified conversion - in practice you'd need full conversion
        return rs_engine.reset_hand(py_state.button)
    
    def profile_advanced_engine_components(self) -> Dict[str, List[float]]:
        """Profile advanced engine components like card dealing, action validation, etc."""
        results = {}
        
        if python_available and PythonEngine is not None:
            print("Profiling Python Advanced Components...")
            self._profile_python_advanced_components(results)
        
        if rust_available and RustEngine is not None and _rs is not None:
            print("Profiling Rust Advanced Components...")
            self._profile_rust_advanced_components(results)
        
        return results
    
    def _profile_python_advanced_components(self, results: Dict[str, List[float]]):
        """Profile advanced Python engine components"""
        if not (python_available and PythonEngine is not None):
            return
            
        engine = PythonEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        
        for i in range(self.num_samples):
            # Time card dealing operations
            with timer("python_card_dealing", results):
                state = engine.reset_hand(i % 6)
                # Deal flop
                if state.round_label == 'Preflop' and hasattr(engine, '_deal_next_street'):
                    engine._deal_next_street(state)
            
            # Time action validation
            state = engine.reset_hand(i % 6)
            if state.next_to_act is not None:
                with timer("python_action_validation", results):
                    legal = engine.legal_actions(state)
                
                # Time individual action checks
                if legal.actions:
                    test_action = legal.actions[0]
                    with timer("python_action_validation_detailed", results):
                        # Simulate the validation logic manually
                        player = state.players[state.next_to_act]
                        owed_amount = engine.owed(state, state.next_to_act)
            
            # Time pot calculation
            with timer("python_pot_calculation", results):
                total_pot = sum(p.bet for p in state.players) + sum(p.cont for p in state.players)
            
            # Time state copying
            with timer("python_state_copy", results):
                import copy
                state_copy = copy.deepcopy(state)
            
            # Time round advancement logic
            if state.next_to_act is None:
                with timer("python_round_advancement", results):
                    done, rewards = engine.advance_round_if_needed(state)
    
    def _profile_rust_advanced_components(self, results: Dict[str, List[float]]):
        """Profile advanced Rust engine components"""
        if not (rust_available and RustEngine is not None and _rs is not None):
            return
            
        engine = RustEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        rs_engine = _rs.NLHEngine(sb=1, bb=2, start_stack=200, num_players=6, seed=42)
        
        for i in range(self.num_samples):
            # Time card dealing via Rust
            with timer("rust_card_dealing_total", results):
                state = engine.reset_hand(i % 6)
                # Note: _deal_next_street might not be exposed in Rust wrapper
            
            # Time direct Rust card dealing
            with timer("rust_card_dealing_internal", results):
                rs_state = rs_engine.reset_hand(i % 6)
                # Direct Rust operations would go here if exposed
            
            # Time action validation
            state = engine.reset_hand(i % 6)
            if state.next_to_act is not None:
                with timer("rust_action_validation_total", results):
                    legal = engine.legal_actions(state)
                
                # Time direct Rust validation
                with timer("rust_action_validation_internal", results):
                    legal_rs = rs_engine.legal_actions_now()
            
            # Time pot calculation
            with timer("rust_pot_calculation", results):
                state = engine.reset_hand(i % 6)
                total_pot = state.pot
            
            # Time round advancement
            if hasattr(rs_engine, 'advance_round_if_needed_now'):
                with timer("rust_round_advancement_internal", results):
                    try:
                        done_rs, rewards_rs, diff = rs_engine.advance_round_if_needed_now()
                    except Exception:
                        pass
    
    def profile_hand_evaluation_performance(self) -> Dict[str, List[float]]:
        """Profile hand evaluation performance across different scenarios"""
        results = {}
        
        print("Profiling Hand Evaluation Performance...")
        
        # Test different board textures and scenarios
        test_cases = self._generate_hand_evaluation_cases()
        
        if python_available:
            self._profile_python_hand_evaluation(results, test_cases)
        
        if rust_available and _rs is not None:
            self._profile_rust_hand_evaluation(results, test_cases)
        
        return results
    
    def _generate_hand_evaluation_cases(self) -> Dict[str, List]:
        """Generate various hand evaluation test cases"""
        cases = {
            'high_card': [],
            'pairs': [],
            'two_pairs': [],
            'trips': [],
            'straights': [],
            'flushes': [],
            'full_houses': [],
            'quads': [],
            'straight_flushes': []
        }
        
        # Generate test cases for each hand type
        # Simplified generation - in practice you'd want more comprehensive cases
        import random
        rng = random.Random(42)
        deck = list(range(52))
        
        for category in cases:
            for _ in range(20):  # 20 cases per category
                rng.shuffle(deck)
                # Take 7 cards (2 hole + 5 board)
                seven_cards = deck[:7]
                cases[category].append(seven_cards)
        
        return cases
    
    def _profile_python_hand_evaluation(self, results: Dict[str, List[float]], test_cases: Dict[str, List]):
        """Profile Python hand evaluation"""
        from nlhe.core.eval import best5_rank_from_7, hand_rank_5
        
        for category, cases in test_cases.items():
            for case in cases:
                with timer(f"python_hand_eval_{category}", results):
                    try:
                        best5_rank_from_7(tuple(case))
                    except Exception:
                        continue
                
                # Also test 5-card evaluation for comparison
                with timer(f"python_hand_eval_5card_{category}", results):
                    try:
                        hand_rank_5(tuple(case[:5]))
                    except Exception:
                        continue
    
    def _profile_rust_hand_evaluation(self, results: Dict[str, List[float]], test_cases: Dict[str, List]):
        """Profile Rust hand evaluation if available"""
        # This would need the Rust hand evaluation exposed
        # For now, we'll test through the engine
        if not (rust_available and RustEngine is not None and _rs is not None):
            return
            
        # Test through engine showdowns
        engine = RustEngine(sb=1, bb=2, start_stack=200, rng=random.Random(42))
        
        for category, cases in test_cases.items():
            for case in cases[:10]:  # Limit for performance
                with timer(f"rust_hand_eval_via_engine_{category}", results):
                    try:
                        # Create a showdown scenario
                        state = engine.reset_hand(0)
                        state.next_to_act = None
                        state.round_label = 'River'
                        state.board = case[2:7]  # 5 board cards
                        state.players[0].hole = (case[0], case[1])  # 2 hole cards
                        # Force evaluation through showdown
                        done, rewards = engine.advance_round_if_needed(state)
                    except Exception:
                        continue
    
    def print_results(self, results: Dict[str, List[float]]):
        """Print detailed timing results"""
        print(f"\n{'='*80}")
        print("COMPONENT TIMING RESULTS")
        print(f"{'='*80}")
        
        # Group results by category
        categories = {
            'Reset Operations': [k for k in results.keys() if 'reset' in k.lower()],
            'Legal Actions': [k for k in results.keys() if 'legal' in k.lower()],
            'Step Operations': [k for k in results.keys() if 'step' in k.lower()],
            'Showdown Operations': [k for k in results.keys() if 'showdown' in k.lower()],
            'Sidepot Calculations': [k for k in results.keys() if 'sidepot' in k.lower()],
            'Hand Evaluation': [k for k in results.keys() if 'hand_eval' in k.lower()],
            'Winner Determination': [k for k in results.keys() if 'winner' in k.lower()],
            'Card Dealing': [k for k in results.keys() if 'card_dealing' in k.lower()],
            'Action Validation': [k for k in results.keys() if 'action_validation' in k.lower()],
            'Pot Calculation': [k for k in results.keys() if 'pot_calculation' in k.lower()],
            'State Operations': [k for k in results.keys() if 'state_copy' in k.lower() or 'round_advancement' in k.lower()],
            'FFI Overhead': [k for k in results.keys() if 'ffi' in k.lower()],
            'Memory Operations': [k for k in results.keys() if any(x in k.lower() for x in ['object', 'creation'])],
            'Conversion Operations': [k for k in results.keys() if 'conversion' in k.lower()],
            'Python Operations': [k for k in results.keys() if k.startswith('python_')],
            'Rust Operations': [k for k in results.keys() if k.startswith('rust_')],
        }
        
        for category, keys in categories.items():
            if not keys:
                continue
                
            print(f"\n{category}:")
            print("-" * 40)
            
            for key in sorted(keys):
                if key in results and results[key]:
                    times = results[key]
                    mean_time = statistics.mean(times)
                    median_time = statistics.median(times)
                    min_time = min(times)
                    max_time = max(times)
                    std_time = statistics.stdev(times) if len(times) > 1 else 0
                    
                    print(f"  {key:30s}: {mean_time:8.4f}ms ± {std_time:6.4f}ms "
                          f"(median: {median_time:6.4f}ms, range: {min_time:6.4f}-{max_time:6.4f}ms, n={len(times)})")
        
        # Calculate overhead percentages
        if 'rust_reset_total' in results and 'rust_reset_internal' in results:
            total_time = statistics.mean(results['rust_reset_total'])
            internal_time = statistics.mean(results['rust_reset_internal'])
            overhead = total_time - internal_time
            overhead_pct = (overhead / total_time) * 100 if total_time > 0 else 0
            print(f"\nReset FFI Overhead: {overhead:.4f}ms ({overhead_pct:.1f}% of total time)")
        
        if 'rust_step_total' in results and 'rust_step_internal' in results:
            total_time = statistics.mean(results['rust_step_total'])
            internal_time = statistics.mean(results['rust_step_internal'])
            overhead = total_time - internal_time
            overhead_pct = (overhead / total_time) * 100 if total_time > 0 else 0
            print(f"Step FFI Overhead:  {overhead:.4f}ms ({overhead_pct:.1f}% of total time)")
    
    def _print_comprehensive_analysis(self, results: Dict[str, List[float]]):
        """Print comprehensive performance analysis with comparisons"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Engine comparison
        python_times = {}
        rust_times = {}
        
        for key, times in results.items():
            if times and key.startswith('python_'):
                base_key = key[7:]  # Remove 'python_' prefix
                python_times[base_key] = statistics.mean(times)
            elif times and key.startswith('rust_') and '_total' in key:
                base_key = key[5:].replace('_total', '')  # Remove 'rust_' prefix and '_total'
                rust_times[base_key] = statistics.mean(times)
        
        # Find common operations for direct comparison
        common_ops = set(python_times.keys()) & set(rust_times.keys())
        if common_ops:
            print("\nDirect Performance Comparison (Python vs Rust):")
            print("-" * 60)
            for op in sorted(common_ops):
                py_time = python_times[op]
                rust_time = rust_times[op]
                speedup = py_time / rust_time if rust_time > 0 else float('inf')
                print(f"  {op:25s}: Python {py_time:8.4f}ms  |  Rust {rust_time:8.4f}ms  |  "
                      f"Speedup: {speedup:6.2f}x")
        
        # Showdown performance analysis
        showdown_categories = ['heads_up', 'sidepot_simple', 'sidepot_complex']
        for category in showdown_categories:
            py_key = f'python_showdown_{category}'
            rust_key = f'rust_showdown_total_{category}'
            
            if py_key in results and rust_key in results:
                py_times = results[py_key]
                rust_times = results[rust_key]
                if py_times and rust_times:
                    py_mean = statistics.mean(py_times)
                    rust_mean = statistics.mean(rust_times)
                    speedup = py_mean / rust_mean if rust_mean > 0 else float('inf')
                    print(f"\nShowdown {category}: Python {py_mean:.4f}ms vs Rust {rust_mean:.4f}ms "
                          f"(Speedup: {speedup:.2f}x)")
        
        # FFI overhead analysis
        ffi_analysis = []
        for key in results.keys():
            if '_total' in key and key.replace('_total', '_internal') in results:
                internal_key = key.replace('_total', '_internal')
                if results[key] and results[internal_key]:
                    total_time = statistics.mean(results[key])
                    internal_time = statistics.mean(results[internal_key])
                    overhead = total_time - internal_time
                    overhead_pct = (overhead / total_time) * 100 if total_time > 0 else 0
                    ffi_analysis.append((key.replace('rust_', '').replace('_total', ''), overhead, overhead_pct))
        
        if ffi_analysis:
            print(f"\nFFI Overhead Analysis:")
            print("-" * 50)
            for op, overhead_ms, overhead_pct in sorted(ffi_analysis):
                print(f"  {op:25s}: {overhead_ms:8.4f}ms ({overhead_pct:5.1f}% overhead)")
        
        # Hand evaluation performance by category
        hand_eval_categories = ['high_card', 'pairs', 'straights', 'flushes', 'full_houses']
        print(f"\nHand Evaluation Performance by Category:")
        print("-" * 55)
        
        for category in hand_eval_categories:
            py_key = f'python_hand_eval_{category}'
            rust_key = f'rust_hand_eval_via_engine_{category}'
            
            if py_key in results:
                py_times = results[py_key]
                if py_times:
                    py_mean = statistics.mean(py_times)
                    print(f"  {category:15s}: Python {py_mean:8.4f}ms", end="")
                    
                    if rust_key in results and results[rust_key]:
                        rust_times = results[rust_key]
                        rust_mean = statistics.mean(rust_times)
                        speedup = py_mean / rust_mean if rust_mean > 0 else float('inf')
                        print(f"  |  Rust {rust_mean:8.4f}ms  |  Speedup: {speedup:6.2f}x")
                    else:
                        print()
        
        # Memory and state operations
        memory_ops = [k for k in results.keys() if 'object_creation' in k or 'state_copy' in k]
        if memory_ops:
            print(f"\nMemory Operations Performance:")
            print("-" * 40)
            for key in sorted(memory_ops):
                if results[key]:
                    mean_time = statistics.mean(results[key])
                    op_name = key.replace('python_', '').replace('rust_', '')
                    engine_type = 'Python' if 'python_' in key else 'Rust'
                    print(f"  {op_name:20s} ({engine_type}): {mean_time:8.4f}ms")


def main():
    """Main profiling function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Rust engine components")
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples for profiling (default: 100)')
    parser.add_argument('--python-only', action='store_true',
                       help='Only profile Python engine')
    parser.add_argument('--rust-only', action='store_true', 
                       help='Only profile Rust engine')
    parser.add_argument('--ffi-only', action='store_true',
                       help='Only profile FFI overhead')
    parser.add_argument('--memory-only', action='store_true',
                       help='Only profile memory operations')
    parser.add_argument('--showdown-only', action='store_true',
                       help='Only profile showdown and sidepot operations')
    parser.add_argument('--advanced-only', action='store_true',
                       help='Only profile advanced engine components')
    parser.add_argument('--hand-eval-only', action='store_true',
                       help='Only profile hand evaluation performance')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run all comprehensive benchmarks (showdown, advanced, hand evaluation)')
    
    args = parser.parse_args()
    
    profiler = ComponentProfiler(args.samples)
    all_results = {}
    
    print("Extended Component-Level Performance Profiler")
    print("=" * 60)
    print(f"Samples per test: {args.samples}")
    print(f"Python available: {python_available}")
    print(f"Rust available:   {rust_available}")
    print()
    
    # Run basic profiling unless specifically excluded
    if not any([args.ffi_only, args.memory_only, args.showdown_only, 
                args.advanced_only, args.hand_eval_only]):
        
        if not args.rust_only and python_available:
            python_results = profiler.profile_python_engine()
            all_results.update(python_results)
        
        if not args.python_only and rust_available:
            rust_results = profiler.profile_rust_engine_detailed()
            all_results.update(rust_results)
    
    # Run FFI overhead profiling
    if (args.ffi_only or args.comprehensive or 
        not any([args.python_only, args.rust_only, args.memory_only, 
                 args.showdown_only, args.advanced_only, args.hand_eval_only])):
        if rust_available:
            ffi_results = profiler.profile_ffi_overhead()
            all_results.update(ffi_results)
    
    # Run memory operations profiling
    if (args.memory_only or args.comprehensive or 
        not any([args.python_only, args.rust_only, args.ffi_only, 
                 args.showdown_only, args.advanced_only, args.hand_eval_only])):
        memory_results = profiler.profile_memory_operations()
        all_results.update(memory_results)
    
    # Run comprehensive showdown and sidepot profiling
    if args.showdown_only or args.comprehensive:
        showdown_results = profiler.profile_showdown_sidepot_detailed()
        all_results.update(showdown_results)
    
    # Run advanced engine components profiling
    if args.advanced_only or args.comprehensive:
        advanced_results = profiler.profile_advanced_engine_components()
        all_results.update(advanced_results)
    
    # Run hand evaluation profiling
    if args.hand_eval_only or args.comprehensive:
        hand_eval_results = profiler.profile_hand_evaluation_performance()
        all_results.update(hand_eval_results)
    
    # Print all results
    profiler.print_results(all_results)
    
    # Print comprehensive analysis
    profiler._print_comprehensive_analysis(all_results)
    
    print(f"\n{'='*80}")
    print("EXTENDED ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("\nKey Performance Insights:")
    print("- Compare 'internal' vs 'total' times to see FFI overhead")
    print("- 'conversion' times show cost of Python ↔ Rust data translation")
    print("- Showdown operations include hand evaluation and sidepot calculations")
    print("- Different hand types show evaluation performance characteristics")
    print("- FFI overhead is the difference between total and internal times")
    print("- Memory operations show object creation and state copying costs")


if __name__ == "__main__":
    main()
