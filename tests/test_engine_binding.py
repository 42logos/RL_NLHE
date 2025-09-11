import nlhe_engine


def test_step_apply_updates_python_state():
    engine = nlhe_engine.NLHEngine(sb=1, bb=2, start_stack=100, seed=0)
    state = engine.reset_hand(button=0)
    actor = state.next_to_act
    assert actor is not None
    done, _ = engine.step_apply_py_raw(state, 2, None)
    assert not done
    assert state.players[actor].bet == state.current_bet


def test_round_advance_resets_bets():
    engine = nlhe_engine.NLHEngine(sb=1, bb=2, start_stack=100, seed=0)
    state = engine.reset_hand(button=0)
    while state.round_label == "Preflop":
        player = state.players[state.next_to_act]
        owe = max(0, state.current_bet - player.bet)
        action = 2 if owe > 0 else 1
        engine.step_apply_py_raw(state, action, None)
    assert state.round_label == "Flop"
    assert all(p.bet == 0 for p in state.players)
