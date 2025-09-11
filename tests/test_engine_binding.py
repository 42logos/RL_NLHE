import importlib.util, importlib.machinery, pathlib
spec = importlib.util.find_spec("nlhe_engine")
suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
lib_path = pathlib.Path(spec.origin).with_name(f"nlhe_engine{suffix}")
spec = importlib.util.spec_from_file_location("nlhe_engine", lib_path)
nlhe_engine = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(nlhe_engine)  # type: ignore[arg-type]


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
