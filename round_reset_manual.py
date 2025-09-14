#!/usr/bin/env python3
"""
Test to verify that round reset behavior works correctly when transitioning between streets.
This test checks that player.bet values are properly reset to 0 when advancing from preflop to flop.
"""

import importlib.util, importlib.machinery, pathlib
spec = importlib.util.find_spec("nlhe_engine")
suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
lib_path = pathlib.Path(spec.origin).with_name(f"nlhe_engine{suffix}")
spec = importlib.util.spec_from_file_location("nlhe_engine", lib_path)
nlhe_engine = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(nlhe_engine)  # type: ignore[arg-type]
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
    
    # Let remaining players act until round completion.
    # Players who owe chips must CALL; others simply CHECK.
    while state.next_to_act is not None and state.round_label == "Preflop":
        idx = state.next_to_act
        owe = state.current_bet - state.players[idx].bet
        # The big blind (or any fully invested player) owes zero chips and must CHECK; CALL would raise ValueError.
        action = 2 if owe > 0 else 1  # 0=FOLD,1=CHECK,2=CALL,3=RAISE_TO
        print(f"Player {idx} taking action {ActionType(action + 1).name}")
        done, rewards = engine.step_apply_py_raw(state, action, None)
        print(f"Player bets: {[p.bet for p in state.players]}")
        assert not done

    print(f"Preflop round finished, next_to_act: {state.next_to_act}")

    # After the final CHECK by the big blind the round should have advanced
    # automatically to the flop, resetting all player bets to zero.
    assert state.round_label == "Flop"
    print("Player bets before verification:", [p.bet for p in state.players])

    # Calling advance_round_if_needed_apply_py should now be a no-op since the
    # round has already advanced. This exercises the helper but does not change
    # the state.
    done, rewards = engine.advance_round_if_needed_apply_py(state)

    print("=== After Round Advance Check ===")
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
