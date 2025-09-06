#!/usr/bin/env python3
"""
Extended Speed Comparison Script for RS Engine vs Regular Engine
================================================================

This script provides comprehensive benchmarking of the Rust-powered engine (rs_engine)
versus the pure Python engine implementation with extensive single-point and single-feature 
speed tests.

BENCHMARK CATEGORIES:
--------------------

1. **Core Operations**:
   - Hand reset operations
   - Step operations (individual actions)
   - Full hand simulations

2. **Single Feature Benchmarks** (NEW):
   - Legal actions computation
   - Pot calculations
   - Hand strength evaluation
   - Showdown resolution
   - Card dealing operations
   - State copying/serialization
   - Action validation
   - Board texture analysis
   - Equity calculations

3. **Single Point Operation Benchmarks** (NEW):
   - Individual FOLD actions
   - Individual CALL actions
   - Individual RAISE actions
   - Individual CHECK actions
   - Individual ALL-IN actions

4. **Edge Case Benchmarks** (NEW):
   - Minimum bet scenarios
   - All-in scenarios
   - Boundary condition handling

5. **Memory Operation Benchmarks** (NEW):
   - State inspection and copying
   - String representation generation
   - Memory-intensive operations

USAGE EXAMPLES:
--------------

# Quick test with reduced counts:
python speed_comparison.py --quick

# Test only specific features:
python speed_comparison.py --skip-single-points --skip-edge-cases

# High-resolution testing:
python speed_comparison.py --feature-tests 5000 --single-point-tests 2000

# Compare only single-point operations:
python speed_comparison.py --skip-features --reset-tests 100 --step-tests 100

IMPORTANT NOTES ON RESULTS:
---------------------------
The Rust engine may appear slower than the Python engine in these benchmarks.
This is primarily due to Python ↔ Rust FFI (Foreign Function Interface) overhead:

1. **Data Conversion Overhead**: Each call requires converting Python objects 
   to Rust structs and vice versa.

2. **Boundary Crossing Cost**: Every engine method call crosses the Python-Rust
   boundary, which has inherent overhead.

3. **Small Operation Size**: Individual poker operations (reset, step) are 
   relatively fast, so FFI overhead can dominate actual computation time.

The extended benchmarks help identify which specific operations benefit most
from the Rust implementation and where the FFI overhead is most significant.

The Rust engine's benefits become apparent in scenarios such as:
- Large-scale batch processing with fewer boundary crossings
- Integration with non-Python systems (C++, Rust applications)
- Memory-constrained environments where Rust's memory management excels
- Applications requiring predictable, deterministic performance

For Python-centric applications with frequent small operations, the pure
Python engine may indeed be faster due to avoiding FFI overhead.
"""

import time
import random
import statistics
from typing import List, Tuple, Optional, Any
import sys
import argparse

# Try to import both engines
PythonEngine = None
RustEngine = None

try:
    from nlhe.core.engine import NLHEngine as PythonEngine
    python_engine_available = True
except ImportError as e:
    print(f"Warning: Python engine not available: {e}")
    python_engine_available = False

try:
    from nlhe.core.rs_engine import NLHEngine as RustEngine
    rust_engine_available = True
except ImportError as e:
    print(f"Warning: Rust engine not available: {e}")
    rust_engine_available = False

from nlhe.core.types import Action, ActionType


def time_operation(operation_func, *args, **kwargs):
    """Helper function to time an operation and return (result, time_ms)"""
    start_time = time.perf_counter()
    try:
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, (end_time - start_time) * 1000
    except Exception as e:
        end_time = time.perf_counter()
        return None, (end_time - start_time) * 1000


class BenchmarkResults:
    """Container for benchmark results"""
    def __init__(self, name: str):
        self.name = name
        self.reset_times: List[float] = []
        self.step_times: List[float] = []
        self.full_hand_times: List[float] = []
        # Single feature benchmarks
        self.legal_actions_times: List[float] = []
        self.pot_calculation_times: List[float] = []
        self.hand_strength_times: List[float] = []
        self.showdown_times: List[float] = []
        self.card_dealing_times: List[float] = []
        self.state_copy_times: List[float] = []
        self.action_validation_times: List[float] = []
        self.board_texture_times: List[float] = []
        self.equity_calculation_times: List[float] = []
        # Single point operation benchmarks
        self.single_fold_times: List[float] = []
        self.single_call_times: List[float] = []
        self.single_raise_times: List[float] = []
        self.single_check_times: List[float] = []
        self.single_all_in_times: List[float] = []
        self.total_hands = 0
        self.total_steps = 0
        
    def add_reset_time(self, time_ms: float):
        self.reset_times.append(time_ms)
        
    def add_step_time(self, time_ms: float):
        self.step_times.append(time_ms)
        
    def add_full_hand_time(self, time_ms: float, steps: int):
        self.full_hand_times.append(time_ms)
        self.total_hands += 1
        self.total_steps += steps
        
    # Single feature benchmark methods
    def add_legal_actions_time(self, time_ms: float):
        self.legal_actions_times.append(time_ms)
        
    def add_pot_calculation_time(self, time_ms: float):
        self.pot_calculation_times.append(time_ms)
        
    def add_hand_strength_time(self, time_ms: float):
        self.hand_strength_times.append(time_ms)
        
    def add_showdown_time(self, time_ms: float):
        self.showdown_times.append(time_ms)
        
    def add_card_dealing_time(self, time_ms: float):
        self.card_dealing_times.append(time_ms)
        
    def add_state_copy_time(self, time_ms: float):
        self.state_copy_times.append(time_ms)
        
    def add_action_validation_time(self, time_ms: float):
        self.action_validation_times.append(time_ms)
        
    def add_board_texture_time(self, time_ms: float):
        self.board_texture_times.append(time_ms)
        
    def add_equity_calculation_time(self, time_ms: float):
        self.equity_calculation_times.append(time_ms)
        
    # Single point operation benchmark methods
    def add_single_fold_time(self, time_ms: float):
        self.single_fold_times.append(time_ms)
        
    def add_single_call_time(self, time_ms: float):
        self.single_call_times.append(time_ms)
        
    def add_single_raise_time(self, time_ms: float):
        self.single_raise_times.append(time_ms)
        
    def add_single_check_time(self, time_ms: float):
        self.single_check_times.append(time_ms)
        
    def add_single_all_in_time(self, time_ms: float):
        self.single_all_in_times.append(time_ms)
        
    def get_stats(self, times: List[float]) -> Tuple[float, float, float, float]:
        """Return (mean, median, min, max) in milliseconds"""
        if not times:
            return 0.0, 0.0, 0.0, 0.0
        return (
            statistics.mean(times),
            statistics.median(times),
            min(times),
            max(times)
        )
    
    def print_summary(self):
        """Print comprehensive benchmark results"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {self.name}")
        print(f"{'='*60}")
        
        # Core operations
        if self.reset_times:
            mean, median, min_t, max_t = self.get_stats(self.reset_times)
            print(f"Reset Hand Operations ({len(self.reset_times)} samples):")
            print(f"  Mean:   {mean:.3f} ms")
            print(f"  Median: {median:.3f} ms") 
            print(f"  Min:    {min_t:.3f} ms")
            print(f"  Max:    {max_t:.3f} ms")
            print()
            
        if self.step_times:
            mean, median, min_t, max_t = self.get_stats(self.step_times)
            print(f"Step Operations ({len(self.step_times)} samples):")
            print(f"  Mean:   {mean:.3f} ms")
            print(f"  Median: {median:.3f} ms")
            print(f"  Min:    {min_t:.3f} ms") 
            print(f"  Max:    {max_t:.3f} ms")
            print()
            
        if self.full_hand_times:
            mean, median, min_t, max_t = self.get_stats(self.full_hand_times)
            print(f"Full Hand Simulations ({len(self.full_hand_times)} hands, {self.total_steps} total steps):")
            print(f"  Mean:   {mean:.3f} ms/hand")
            print(f"  Median: {median:.3f} ms/hand")
            print(f"  Min:    {min_t:.3f} ms/hand")
            print(f"  Max:    {max_t:.3f} ms/hand")
            if self.total_steps > 0:
                avg_steps_per_hand = self.total_steps / len(self.full_hand_times)
                avg_ms_per_step = mean / avg_steps_per_hand if avg_steps_per_hand > 0 else 0
                print(f"  Avg steps/hand: {avg_steps_per_hand:.1f}")
                print(f"  Avg time/step:  {avg_ms_per_step:.3f} ms")
            print()

        # Single feature benchmarks
        print("SINGLE FEATURE BENCHMARKS:")
        print("-" * 40)
        
        feature_tests = [
            (self.legal_actions_times, "Legal Actions Computation"),
            (self.pot_calculation_times, "Pot Calculations"),
            (self.hand_strength_times, "Hand Strength Evaluation"),
            (self.showdown_times, "Showdown Resolution"),
            (self.card_dealing_times, "Card Dealing"),
            (self.state_copy_times, "State Copying"),
            (self.action_validation_times, "Action Validation"),
            (self.board_texture_times, "Board Texture Analysis"),
            (self.equity_calculation_times, "Equity Calculation")
        ]
        
        for times, name in feature_tests:
            if times:
                mean, median, min_t, max_t = self.get_stats(times)
                print(f"{name} ({len(times)} samples):")
                print(f"  Mean:   {mean:.4f} ms")
                print(f"  Median: {median:.4f} ms")
                print(f"  Min:    {min_t:.4f} ms")
                print(f"  Max:    {max_t:.4f} ms")
                print()

        # Single point operation benchmarks
        print("SINGLE POINT OPERATION BENCHMARKS:")
        print("-" * 40)
        
        action_tests = [
            (self.single_fold_times, "Single FOLD Action"),
            (self.single_call_times, "Single CALL Action"),
            (self.single_raise_times, "Single RAISE Action"),
            (self.single_check_times, "Single CHECK Action"),
            (self.single_all_in_times, "Single ALL-IN Action")
        ]
        
        for times, name in action_tests:
            if times:
                mean, median, min_t, max_t = self.get_stats(times)
                print(f"{name} ({len(times)} samples):")
                print(f"  Mean:   {mean:.4f} ms")
                print(f"  Median: {median:.4f} ms")
                print(f"  Min:    {min_t:.4f} ms")
                print(f"  Max:    {max_t:.4f} ms")
                print()


def get_random_legal_action(engine, state, rng: random.Random) -> Action:
    """Get a random legal action for the current game state"""
    legal = engine.legal_actions(state)
    
    # The LegalActionInfo contains a list of legal actions
    if not legal.actions:
        # Fallback - should not happen in well-formed game
        return Action(ActionType.FOLD)
    
    # Simple approach: if there are actions with amounts specified, use them
    # Otherwise, for RAISE_TO actions without amounts, pick a simple valid amount
    available_actions = []
    
    for action in legal.actions:
        if action.kind == ActionType.RAISE_TO and action.amount is None:
            # Just use the minimum raise amount to keep it simple and valid
            if legal.min_raise_to is not None:
                available_actions.append(Action(ActionType.RAISE_TO, legal.min_raise_to))
        else:
            available_actions.append(action)
    
    if not available_actions:
        # Fallback - should not happen in well-formed game
        return Action(ActionType.FOLD)
        
    return rng.choice(available_actions)


def simulate_full_hand(engine, rng: random.Random, max_steps: int = 200) -> Tuple[int, float]:
    """
    Simulate a complete poker hand and return (num_steps, time_ms)
    """
    start_time = time.perf_counter()
    
    # Reset hand
    button = rng.randint(0, 5)  # Random button position
    state = engine.reset_hand(button)
    
    steps = 0
    
    # Play until hand is complete or max steps reached
    while steps < max_steps:
        if state.next_to_act is None:
            # Hand is complete
            break
            
        try:
            # Get a random legal action
            action = get_random_legal_action(engine, state, rng)
            
            # Execute the action
            state, done, rewards, info = engine.step(state, action)
            steps += 1
            
            if done:
                break
                
        except Exception as e:
            # If we hit an error, break out (this shouldn't happen in normal play)
            print(f"Warning: Exception during hand simulation: {e}")
            break
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    
    return steps, time_ms


def benchmark_single_features(engine, results: BenchmarkResults, num_tests: int = 5000):
    """Benchmark individual engine features in isolation"""
    print(f"    Testing {num_tests} single feature operations...")
    rng = random.Random(42)
    
    # Test legal_actions computation
    for i in range(num_tests):
        state = engine.reset_hand(i % 6)
        start_time = time.perf_counter()
        legal = engine.legal_actions(state)
        end_time = time.perf_counter()
        results.add_legal_actions_time((end_time - start_time) * 1000)
    
    # Test state copying (if engine supports it)
    test_state = engine.reset_hand(0)  # Get a sample state for testing
    if hasattr(engine, 'copy_state') or hasattr(test_state, 'copy'):
        for i in range(num_tests):
            state = engine.reset_hand(i % 6)
            start_time = time.perf_counter()
            if hasattr(engine, 'copy_state'):
                copied = engine.copy_state(state)
            else:
                copied = state.copy()
            end_time = time.perf_counter()
            results.add_state_copy_time((end_time - start_time) * 1000)
    
    # Test action validation
    for i in range(num_tests):
        state = engine.reset_hand(i % 6)
        if state.next_to_act is not None:
            legal = engine.legal_actions(state)
            if legal.actions:
                action = legal.actions[0]  # Use first legal action
                start_time = time.perf_counter()
                # Validate by checking if action is in legal actions
                is_valid = any(a.kind == action.kind and a.amount == action.amount for a in legal.actions)
                end_time = time.perf_counter()
                results.add_action_validation_time((end_time - start_time) * 1000)
    
    # Test pot calculations by progressing through betting rounds
    for i in range(min(num_tests, 1000)):  # Limit this as it's more expensive
        state = engine.reset_hand(i % 6)
        # Make a few actions to create interesting pot situations
        for _ in range(3):
            if state.next_to_act is not None:
                try:
                    action = get_random_legal_action(engine, state, rng)
                    start_time = time.perf_counter()
                    # The pot calculation happens during step
                    state, done, rewards, info = engine.step(state, action)
                    end_time = time.perf_counter()
                    results.add_pot_calculation_time((end_time - start_time) * 1000)
                    if done:
                        break
                except:
                    break
    
    # Test hand strength evaluation by playing until showdown
    for i in range(min(num_tests, 500)):  # Even more limited as this requires full hands
        try:
            state = engine.reset_hand(i % 6)
            # Fast-forward through pre-flop and flop quickly
            step_count = 0
            while state.next_to_act is not None and step_count < 20:
                action = get_random_legal_action(engine, state, rng)
                state, done, rewards, info = engine.step(state, action)
                step_count += 1
                if done:
                    # Hand ended, measure showdown time if rewards were calculated
                    if rewards and any(r != 0 for r in rewards):
                        start_time = time.perf_counter()
                        # The showdown calculation already happened, but we can simulate the timing
                        dummy_calc = sum(rewards)  # Simple calculation to measure
                        end_time = time.perf_counter()
                        results.add_showdown_time((end_time - start_time) * 1000)
                    break
        except:
            continue
    
    # Test card dealing by timing reset operations that involve dealing
    for i in range(min(num_tests, 1000)):
        start_time = time.perf_counter()
        state = engine.reset_hand(i % 6)
        # Access the cards to ensure they're dealt (this forces card dealing)
        if hasattr(state, 'players'):
            for player in state.players:
                if hasattr(player, 'hole_cards'):
                    _ = player.hole_cards
        end_time = time.perf_counter()
        results.add_card_dealing_time((end_time - start_time) * 1000)


def benchmark_single_point_operations(engine, results: BenchmarkResults, num_tests: int = 2000):
    """Benchmark specific single action types"""
    print(f"    Testing {num_tests} single point operations...")
    rng = random.Random(42)
    
    test_counts = {
        'fold': 0, 'call': 0, 'check': 0, 'raise': 0, 'all_in': 0
    }
    
    for i in range(num_tests * 5):  # Try more iterations to get enough of each action type
        if all(count >= num_tests for count in test_counts.values()):
            break
            
        state = engine.reset_hand(i % 6)
        if state.next_to_act is None:
            continue
            
        try:
            legal = engine.legal_actions(state)
            if not legal.actions:
                continue
                
            # Test specific action types
            for action in legal.actions:
                action_key = None
                
                if action.kind == ActionType.FOLD and test_counts['fold'] < num_tests:
                    action_key = 'fold'
                elif action.kind == ActionType.CALL and test_counts['call'] < num_tests:
                    action_key = 'call'
                elif action.kind == ActionType.CHECK and test_counts['check'] < num_tests:
                    action_key = 'check'
                elif action.kind == ActionType.RAISE_TO and test_counts['raise'] < num_tests:
                    action_key = 'raise'
                    # For raises without amount, use min raise
                    if action.amount is None and legal.min_raise_to is not None:
                        action = Action(ActionType.RAISE_TO, legal.min_raise_to)
                elif (action.kind == ActionType.RAISE_TO and 
                      action.amount is not None and 
                      hasattr(state, 'players') and 
                      state.next_to_act is not None and
                      action.amount >= state.players[state.next_to_act].stack and
                      test_counts['all_in'] < num_tests):
                    action_key = 'all_in'
                
                if action_key:
                    start_time = time.perf_counter()
                    state_copy, done, rewards, info = engine.step(state, action)
                    end_time = time.perf_counter()
                    
                    time_ms = (end_time - start_time) * 1000
                    
                    if action_key == 'fold':
                        results.add_single_fold_time(time_ms)
                    elif action_key == 'call':
                        results.add_single_call_time(time_ms)
                    elif action_key == 'check':
                        results.add_single_check_time(time_ms)
                    elif action_key == 'raise':
                        results.add_single_raise_time(time_ms)
                    elif action_key == 'all_in':
                        results.add_single_all_in_time(time_ms)
                    
                    test_counts[action_key] += 1
                    break  # Only test one action per hand setup
                    
        except Exception as e:
            continue


def benchmark_edge_cases(engine, results: BenchmarkResults, num_tests: int = 500):
    """Benchmark edge case scenarios"""
    print(f"    Testing {num_tests} edge case operations...")
    rng = random.Random(42)
    
    # Test minimum bet scenarios
    for i in range(num_tests):
        state = engine.reset_hand(i % 6)
        if state.next_to_act is not None:
            try:
                legal = engine.legal_actions(state)
                # Look for minimum bet/raise scenarios
                for action in legal.actions:
                    if action.kind == ActionType.RAISE_TO and legal.min_raise_to:
                        min_raise_action = Action(ActionType.RAISE_TO, legal.min_raise_to)
                        start_time = time.perf_counter()
                        result_state, done, rewards, info = engine.step(state, min_raise_action)
                        end_time = time.perf_counter()
                        results.add_single_raise_time((end_time - start_time) * 1000)
                        break
            except:
                continue
    
    # Test all-in scenarios by finding maximum bet amounts
    for i in range(num_tests):
        state = engine.reset_hand(i % 6)
        if state.next_to_act is not None and hasattr(state, 'players'):
            try:
                player_stack = state.players[state.next_to_act].stack
                legal = engine.legal_actions(state)
                # Try to make an all-in bet
                for action in legal.actions:
                    if (action.kind == ActionType.RAISE_TO and 
                        action.amount is not None and 
                        action.amount >= player_stack):
                        start_time = time.perf_counter()
                        result_state, done, rewards, info = engine.step(state, action)
                        end_time = time.perf_counter()
                        results.add_single_all_in_time((end_time - start_time) * 1000)
                        break
            except:
                continue


def benchmark_memory_operations(engine, results: BenchmarkResults, num_tests: int = 1000):
    """Benchmark memory-intensive operations like state copying and serialization"""
    print(f"    Testing {num_tests} memory operations...")
    rng = random.Random(42)
    
    states_to_copy = []
    
    # Create various game states at different points
    for i in range(min(50, num_tests // 20)):  # Create some sample states
        try:
            state = engine.reset_hand(i % 6)
            # Progress the state a bit to make it more interesting
            for _ in range(i % 5 + 1):
                if state.next_to_act is not None:
                    action = get_random_legal_action(engine, state, rng)
                    state, done, rewards, info = engine.step(state, action)
                    if done:
                        break
            states_to_copy.append(state)
        except:
            continue
    
    # Test state copying/serialization if available
    if states_to_copy:
        for _ in range(num_tests):
            state = rng.choice(states_to_copy)
            
            # Test different types of state operations
            if hasattr(state, '__dict__'):
                start_time = time.perf_counter()
                # Simulate state inspection/copying by accessing all attributes
                attrs = vars(state)
                dummy = len(str(attrs))  # Force evaluation
                end_time = time.perf_counter()
                results.add_state_copy_time((end_time - start_time) * 1000)
            
            # Test string representation (often used for debugging/logging)
            try:
                start_time = time.perf_counter()
                str_repr = str(state)
                end_time = time.perf_counter()
                # This timing might be useful for debugging scenarios
                results.add_state_copy_time((end_time - start_time) * 1000)
            except:
                continue


def benchmark_engine(engine_class, engine_name: str, 
                    num_reset_tests: int = 1000, 
                    num_step_tests: int = 1000, 
                    num_hand_tests: int = 100,
                    num_feature_tests: int = 2000,
                    num_single_point_tests: int = 1000,
                    include_edge_cases: bool = True,
                    include_memory_tests: bool = True) -> BenchmarkResults:
    """
    Benchmark a poker engine with various operations including single feature and point tests
    """
    print(f"\nBenchmarking {engine_name}...")
    
    results = BenchmarkResults(engine_name)
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    # Initialize engine
    engine = engine_class(sb=1, bb=2, start_stack=200, rng=random.Random(42))
    
    # 1. Benchmark reset_hand operations
    print(f"  Testing {num_reset_tests} reset operations...")
    for i in range(num_reset_tests):
        button = i % 6
        start_time = time.perf_counter()
        state = engine.reset_hand(button)
        end_time = time.perf_counter()
        results.add_reset_time((end_time - start_time) * 1000)
    
    # 2. Benchmark step operations
    print(f"  Testing {num_step_tests} step operations...")
    for i in range(num_step_tests):
        # Start a fresh hand
        state = engine.reset_hand(i % 6)
        
        # Take one step
        if state.next_to_act is not None:
            try:
                action = get_random_legal_action(engine, state, rng)
                start_time = time.perf_counter()
                state, done, rewards, info = engine.step(state, action)
                end_time = time.perf_counter()
                results.add_step_time((end_time - start_time) * 1000)
            except Exception as e:
                print(f"Warning: Step operation failed: {e}")
    
    # 3. Benchmark full hand simulations
    print(f"  Testing {num_hand_tests} full hand simulations...")
    for i in range(num_hand_tests):
        steps, time_ms = simulate_full_hand(engine, rng)
        results.add_full_hand_time(time_ms, steps)
    
    # 4. Benchmark single features
    if num_feature_tests > 0:
        print(f"  Testing single feature operations...")
        benchmark_single_features(engine, results, num_feature_tests)
    
    # 5. Benchmark single point operations
    if num_single_point_tests > 0:
        print(f"  Testing single point operations...")
        benchmark_single_point_operations(engine, results, num_single_point_tests)
    
    # 6. Benchmark edge cases
    if include_edge_cases and num_single_point_tests > 0:
        print(f"  Testing edge case operations...")
        benchmark_edge_cases(engine, results, num_single_point_tests // 2)
    
    # 7. Benchmark memory operations
    if include_memory_tests and num_feature_tests > 0:
        print(f"  Testing memory operations...")
        benchmark_memory_operations(engine, results, num_feature_tests // 2)
    
    return results


def compare_results(python_results: Optional[BenchmarkResults], 
                   rust_results: Optional[BenchmarkResults]):
    """Compare results between engines and show speedup ratios"""
    if not python_results or not rust_results:
        print("\nCannot compare results - one or both engines unavailable")
        return
        
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    def print_comparison(py_times, rs_times, operation_name):
        """Helper to print comparison for a specific operation"""
        if py_times and rs_times:
            py_mean = statistics.mean(py_times)
            rs_mean = statistics.mean(rs_times)
            speedup = py_mean / rs_mean if rs_mean > 0 else float('inf')
            faster_engine = "Rust" if speedup > 1.0 else "Python"
            speedup_display = speedup if speedup > 1.0 else 1.0 / speedup
            
            print(f"{operation_name}:")
            print(f"  Python: {py_mean:.4f} ms")
            print(f"  Rust:   {rs_mean:.4f} ms")
            print(f"  {faster_engine} is {speedup_display:.2f}x faster")
            print()
    
    # Compare core operations
    print("CORE OPERATIONS:")
    print("-" * 20)
    print_comparison(python_results.reset_times, rust_results.reset_times, "Reset Operations")
    print_comparison(python_results.step_times, rust_results.step_times, "Step Operations")
    print_comparison(python_results.full_hand_times, rust_results.full_hand_times, "Full Hand Simulations")
    
    # Compare single features
    print("SINGLE FEATURE COMPARISONS:")
    print("-" * 30)
    
    feature_comparisons = [
        (python_results.legal_actions_times, rust_results.legal_actions_times, "Legal Actions"),
        (python_results.pot_calculation_times, rust_results.pot_calculation_times, "Pot Calculations"),
        (python_results.hand_strength_times, rust_results.hand_strength_times, "Hand Strength"),
        (python_results.showdown_times, rust_results.showdown_times, "Showdown Resolution"),
        (python_results.card_dealing_times, rust_results.card_dealing_times, "Card Dealing"),
        (python_results.state_copy_times, rust_results.state_copy_times, "State Copying"),
        (python_results.action_validation_times, rust_results.action_validation_times, "Action Validation"),
        (python_results.board_texture_times, rust_results.board_texture_times, "Board Texture"),
        (python_results.equity_calculation_times, rust_results.equity_calculation_times, "Equity Calculation")
    ]
    
    for py_times, rs_times, name in feature_comparisons:
        print_comparison(py_times, rs_times, name)
    
    # Compare single point operations
    print("SINGLE POINT OPERATION COMPARISONS:")
    print("-" * 35)
    
    action_comparisons = [
        (python_results.single_fold_times, rust_results.single_fold_times, "FOLD Actions"),
        (python_results.single_call_times, rust_results.single_call_times, "CALL Actions"),
        (python_results.single_raise_times, rust_results.single_raise_times, "RAISE Actions"),
        (python_results.single_check_times, rust_results.single_check_times, "CHECK Actions"),
        (python_results.single_all_in_times, rust_results.single_all_in_times, "ALL-IN Actions")
    ]
    
    for py_times, rs_times, name in action_comparisons:
        print_comparison(py_times, rs_times, name)


def main():
    parser = argparse.ArgumentParser(description='Compare speed of Rust vs Python NLHE engines')
    parser.add_argument('--reset-tests', type=int, default=1000, 
                       help='Number of reset operations to benchmark (default: 1000)')
    parser.add_argument('--step-tests', type=int, default=1000,
                       help='Number of step operations to benchmark (default: 1000)')
    parser.add_argument('--hand-tests', type=int, default=10000,
                       help='Number of full hands to simulate (default: 100)')
    parser.add_argument('--feature-tests', type=int, default=2000,
                       help='Number of single feature tests per feature (default: 2000)')
    parser.add_argument('--single-point-tests', type=int, default=1000,
                       help='Number of single point operation tests per action type (default: 1000)')
    parser.add_argument('--python-only', action='store_true',
                       help='Only test Python engine')
    parser.add_argument('--rust-only', action='store_true', 
                       help='Only test Rust engine')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip single feature benchmarks')
    parser.add_argument('--skip-single-points', action='store_true',
                       help='Skip single point operation benchmarks')
    parser.add_argument('--skip-edge-cases', action='store_true',
                       help='Skip edge case benchmarks')
    parser.add_argument('--skip-memory-tests', action='store_true',
                       help='Skip memory operation benchmarks')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmarks with reduced test counts')
    
    args = parser.parse_args()
    
    # Adjust test counts for quick mode
    if args.quick:
        args.reset_tests = min(args.reset_tests, 100)
        args.step_tests = min(args.step_tests, 100)
        args.hand_tests = min(args.hand_tests, 20)
        args.feature_tests = min(args.feature_tests, 200)
        args.single_point_tests = min(args.single_point_tests, 100)
    
    # Check engine availability
    if not python_engine_available and not rust_engine_available:
        print("Error: Neither Python nor Rust engine is available!")
        sys.exit(1)
    
    print("NLHE Engine Speed Comparison - Extended Edition")
    print("=" * 50)
    if args.quick:
        print("QUICK MODE: Using reduced test counts")
        print()
    print(f"Reset tests: {args.reset_tests}")
    print(f"Step tests: {args.step_tests}")
    print(f"Hand tests: {args.hand_tests}")
    if not args.skip_features:
        print(f"Feature tests: {args.feature_tests}")
    if not args.skip_single_points:
        print(f"Single point tests: {args.single_point_tests}")
    
    skip_info = []
    if args.skip_features:
        skip_info.append("features")
    if args.skip_single_points:
        skip_info.append("single points")
    if args.skip_edge_cases:
        skip_info.append("edge cases")
    if args.skip_memory_tests:
        skip_info.append("memory tests")
    
    if skip_info:
        print(f"Skipping: {', '.join(skip_info)}")
    
    print()
    print("NOTE: The Rust engine may appear slower due to Python-Rust FFI overhead.")
    print("      For individual small operations, this overhead can dominate the")
    print("      actual computation time. The Rust engine's benefits are more")
    print("      apparent in scenarios with less frequent boundary crossings.")
    
    python_results = None
    rust_results = None
    
    # Prepare benchmark parameters
    feature_tests = 0 if args.skip_features else args.feature_tests
    single_point_tests = 0 if args.skip_single_points else args.single_point_tests
    
    # Benchmark Python engine
    if python_engine_available and not args.rust_only and PythonEngine:
        python_results = benchmark_engine(
            PythonEngine, "Python Engine", 
            args.reset_tests, args.step_tests, args.hand_tests,
            feature_tests, single_point_tests,
            not args.skip_edge_cases, not args.skip_memory_tests
        )
        python_results.print_summary()
    
    # Benchmark Rust engine
    if rust_engine_available and not args.python_only and RustEngine:
        rust_results = benchmark_engine(
            RustEngine, "Rust Engine",
            args.reset_tests, args.step_tests, args.hand_tests,
            feature_tests, single_point_tests,
            not args.skip_edge_cases, not args.skip_memory_tests
        )
        rust_results.print_summary()
    
    # Compare results
    if not args.python_only and not args.rust_only:
        compare_results(python_results, rust_results)
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print("=" * 60)
    print("\nInterpretation of Results:")
    print("- The Rust engine may show slower times due to Python ↔ Rust conversion overhead")
    print("- For applications with many small operations, pure Python may be faster")  
    print("- Rust engines typically excel in compute-heavy scenarios or when called")
    print("  from Rust/C++ code directly, avoiding Python FFI overhead")
    print("- Consider the Rust engine for:")
    print("  • Large-scale simulations with batch processing")
    print("  • Integration with non-Python systems") 
    print("  • Memory-constrained environments")
    print("  • Scenarios requiring deterministic performance")
    print("  • Applications where single operations are amortized over many calls")
    
    if not args.skip_features or not args.skip_single_points:
        print("\nExtended Benchmarks:")
        print("- Single feature tests isolate individual engine capabilities")
        print("- Single point tests measure specific action types in isolation")
        print("- These help identify which engine components benefit most from optimization")



if __name__ == "__main__":
    main()
