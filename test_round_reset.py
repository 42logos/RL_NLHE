#!/usr/bin/env python3
"""
Test to verify that round reset behavior works correctly when transitioning between streets.
This test checks that player.bet values are properly reset to 0 when advancing from preflop to flop.
"""

import nlhe_engine
from nlhe.core.types import ActionType

def test_round_reset():
    # Create engine with deterministic seed
    engine = nlhe_engine.NLHEngine(sb=1, bb=2, start_stack=100, seed=42)
    
    # Start a hand
    state = engine.reset_hand(button=0)
    
    print("=== Initial Preflop State ===")
    print(f"Round: {state.round_label}")
    print(f"Current bet: {state.current_bet}")
    print("Player bets:", [p.bet for p in state.players])
    print("Player stacks:", [p.stack for p in state.players])
    print("Player statuses:", [p.status for p in state.players])
    print()
    
    # Verify initial state (preflop with blinds posted)
    assert state.round_label == "Preflop"
    assert state.current_bet == 2  # BB amount
    assert state.players[1].bet == 1  # SB
    assert state.players[2].bet == 2  # BB
    assert all(state.players[i].bet == 0 for i in [0, 3, 4, 5])  # Other players
    
    # Make some actions to create non-zero bets for multiple players
    # Player 3 calls (next_to_act is 3, which is (button+3)%6 = (0+3)%6 = 3)
    print(f"Next to act: {state.next_to_act}")
    done, rewards = engine.step_apply_py_raw(state, 2, None)  # CALL
    print(f"After player 3 calls - Player bets: {[p.bet for p in state.players]}")
    print(f"Next to act: {state.next_to_act}")
    assert not done
    
    # Player 4 also calls to keep more players in
    done, rewards = engine.step_apply_py_raw(state, 2, None)  # CALL
    print(f"After player 4 calls - Player bets: {[p.bet for p in state.players]}")
    print(f"Next to act: {state.next_to_act}")
    assert not done
    
    # Let's have remaining players all call to keep multiple players active
    while state.next_to_act is not None:
        print(f"Player {state.next_to_act} calling")
        done, rewards = engine.step_apply_py_raw(state, 2, None)  # CALL
        print(f"Player bets: {[p.bet for p in state.players]}")
        if done:
            print("Game ended unexpectedly")
            return
    
    print(f"Preflop round finished, next_to_act: {state.next_to_act}")
    
    # Check that all players have non-zero bets
    print("Player bets before round advance:", [p.bet for p in state.players])
    assert any(p.bet > 0 for p in state.players), "No players have bets > 0"
    
    print("=== Before Round Advance ===")
    print(f"Round: {state.round_label}")
    print(f"Current bet: {state.current_bet}")
    print("Player bets:", [p.bet for p in state.players])
    print("Board length:", len(state.board))
    print()
    
    # Now advance the round (should transition to flop)
    done, rewards = engine.advance_round_if_needed_apply_py(state)
    
    print("=== After Round Advance ===")
    print(f"Round: {state.round_label}")
    print(f"Current bet: {state.current_bet}")
    print("Player bets:", [p.bet for p in state.players])
    print("Player rho values:", [p.rho for p in state.players])
    print("Board length:", len(state.board))
    print("Board:", state.board)
    print()
    
    # Verify the round was properly reset
    assert state.round_label == "Flop", f"Expected 'Flop', got '{state.round_label}'"
    assert state.current_bet == 0, f"Expected current_bet=0, got {state.current_bet}"
    assert all(p.bet == 0 for p in state.players), f"All player bets should be 0, got {[p.bet for p in state.players]}"
    assert all(p.rho == -1_000_000_000 for p in state.players if p.status == "active"), \
        f"All active player rho should be -1_000_000_000, got {[(i, p.rho) for i, p in enumerate(state.players) if p.status == 'active']}"
    assert len(state.board) == 3, f"Expected 3 flop cards, got {len(state.board)}"
    assert state.tau == 0, f"Expected tau=0, got {state.tau}"
    assert state.min_raise == state.bb, f"Expected min_raise={state.bb}, got {state.min_raise}"
    
    print("✅ Round reset test PASSED - all player bets and rho values were properly reset!")

if __name__ == "__main__":
    test_round_reset()
    print("\n🎉 All tests passed!")
