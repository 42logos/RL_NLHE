import nlhe.core.rs_engine as rs
from nlhe.core.types import Action, ActionType

def test_step_and_round_reset_wrapper():
    eng = rs.NLHEngine(sb=1, bb=2, start_stack=100)
    state = eng.reset_hand(button=0)
    assert state.round_label == "Preflop"
    # player 3 calls, player 4 calls
    eng.step(state, Action(ActionType.CALL))
    assert state.players[3].bet == 2
    eng.step(state, Action(ActionType.CALL))
    assert state.players[4].bet == 2
    # let remaining players act
    while state.next_to_act is not None and state.round_label == "Preflop":
        idx = state.next_to_act
        owe = state.current_bet - state.players[idx].bet
        act = Action(ActionType.CALL if owe > 0 else ActionType.CHECK)
        eng.step(state, act)
    # round should have advanced
    assert state.round_label == "Flop"
    assert len(state.board) == 3
    assert all(p.bet == 0 for p in state.players)
