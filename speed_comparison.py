#!/usr/bin/env python3
"""
Speed Comparison Script for RS Engine vs Regular Engine
======================================================

This script compares the performance of the Rust-powered engine (rs_engine)
versus the pure Python engine implementation by running multiple poker hands
and measuring execution times.

The benchmark includes:
- Hand reset operations
- Full hand simulations with random actions
- Action validation and step operations

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


class BenchmarkResults:
    """Container for benchmark results"""
    def __init__(self, name: str):
        self.name = name
        self.reset_times: List[float] = []
        self.step_times: List[float] = []
        self.full_hand_times: List[float] = []
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


def benchmark_engine(engine_class, engine_name: str, num_reset_tests: int = 1000, 
                    num_step_tests: int = 1000, num_hand_tests: int = 100) -> BenchmarkResults:
    """
    Benchmark a poker engine with various operations
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
    
    # Compare reset operations
    if python_results.reset_times and rust_results.reset_times:
        py_mean = statistics.mean(python_results.reset_times)
        rs_mean = statistics.mean(rust_results.reset_times)
        speedup = py_mean / rs_mean if rs_mean > 0 else float('inf')
        print(f"Reset Operations:")
        print(f"  Python: {py_mean:.3f} ms")
        print(f"  Rust:   {rs_mean:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x faster")
        print()
    
    # Compare step operations  
    if python_results.step_times and rust_results.step_times:
        py_mean = statistics.mean(python_results.step_times)
        rs_mean = statistics.mean(rust_results.step_times)
        speedup = py_mean / rs_mean if rs_mean > 0 else float('inf')
        print(f"Step Operations:")
        print(f"  Python: {py_mean:.3f} ms") 
        print(f"  Rust:   {rs_mean:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x faster")
        print()
    
    # Compare full hand simulations
    if python_results.full_hand_times and rust_results.full_hand_times:
        py_mean = statistics.mean(python_results.full_hand_times)
        rs_mean = statistics.mean(rust_results.full_hand_times)
        speedup = py_mean / rs_mean if rs_mean > 0 else float('inf')
        print(f"Full Hand Simulations:")
        print(f"  Python: {py_mean:.3f} ms/hand")
        print(f"  Rust:   {rs_mean:.3f} ms/hand") 
        print(f"  Speedup: {speedup:.1f}x faster")
        print()


def main():
    parser = argparse.ArgumentParser(description='Compare speed of Rust vs Python NLHE engines')
    parser.add_argument('--reset-tests', type=int, default=10000, 
                       help='Number of reset operations to benchmark (default: 1000)')
    parser.add_argument('--step-tests', type=int, default=10000,
                       help='Number of step operations to benchmark (default: 1000)')
    parser.add_argument('--hand-tests', type=int, default=10000,
                       help='Number of full hands to simulate (default: 100)')
    parser.add_argument('--python-only', action='store_true',
                       help='Only test Python engine')
    parser.add_argument('--rust-only', action='store_true', 
                       help='Only test Rust engine')
    
    args = parser.parse_args()
    
    # Check engine availability
    if not python_engine_available and not rust_engine_available:
        print("Error: Neither Python nor Rust engine is available!")
        sys.exit(1)
    
    print("NLHE Engine Speed Comparison")
    print("=" * 40)
    print(f"Reset tests: {args.reset_tests}")
    print(f"Step tests: {args.step_tests}")
    print(f"Hand tests: {args.hand_tests}")
    print()
    print("NOTE: The Rust engine may appear slower due to Python-Rust FFI overhead.")
    print("      For individual small operations, this overhead can dominate the")
    print("      actual computation time. The Rust engine's benefits are more")
    print("      apparent in scenarios with less frequent boundary crossings.")
    
    python_results = None
    rust_results = None
    
    # Benchmark Python engine
    if python_engine_available and not args.rust_only and PythonEngine:
        python_results = benchmark_engine(
            PythonEngine, "Python Engine", 
            args.reset_tests, args.step_tests, args.hand_tests
        )
        python_results.print_summary()
    
    # Benchmark Rust engine
    if rust_engine_available and not args.python_only and RustEngine:
        rust_results = benchmark_engine(
            RustEngine, "Rust Engine",
            args.reset_tests, args.step_tests, args.hand_tests
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
    print("  • Large-scale simulations")
    print("  • Integration with non-Python systems") 
    print("  • Memory-constrained environments")
    print("  • Scenarios requiring deterministic performance")


if __name__ == "__main__":
    main()
