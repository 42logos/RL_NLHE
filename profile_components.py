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
    
    args = parser.parse_args()
    
    profiler = ComponentProfiler(args.samples)
    all_results = {}
    
    print("Component-Level Performance Profiler")
    print("=" * 50)
    print(f"Samples per test: {args.samples}")
    print(f"Python available: {python_available}")
    print(f"Rust available:   {rust_available}")
    print()
    
    if not args.rust_only and not args.ffi_only and not args.memory_only and python_available:
        python_results = profiler.profile_python_engine()
        all_results.update(python_results)
    
    if not args.python_only and not args.ffi_only and not args.memory_only and rust_available:
        rust_results = profiler.profile_rust_engine_detailed()
        all_results.update(rust_results)
    
    if not args.python_only and not args.rust_only and not args.memory_only and rust_available:
        ffi_results = profiler.profile_ffi_overhead()
        all_results.update(ffi_results)
    
    if not args.python_only and not args.rust_only and not args.ffi_only:
        memory_results = profiler.profile_memory_operations()
        all_results.update(memory_results)
    
    # Print all results
    profiler.print_results(all_results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("\nKey Insights:")
    print("- Compare 'internal' vs 'total' times to see FFI overhead")
    print("- 'conversion' times show cost of Python ↔ Rust data translation")
    print("- FFI overhead is the difference between total and internal times")
    print("- Memory operations show object creation costs")


if __name__ == "__main__":
    main()
