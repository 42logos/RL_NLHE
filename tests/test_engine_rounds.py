import random
import sys
import types

import pytest

# Only create stub if nlhe_engine is not already available
if "nlhe_engine" not in sys.modules:
    try:
        import nlhe_engine
    except ImportError:
        # Stub out optional Rust evaluator so engine can import
        _stub = types.ModuleType("nlhe_engine")
        sys.modules.setdefault("nlhe_engine", _stub)

from nlhe.core.engine import NLHEngine
from nlhe.core.types import Action, ActionType


def test_all_but_one_fold_terminates_hand():
    eng = NLHEngine(rng=random.Random(0))
    s = eng.reset_hand(button=0)

    # Fold everyone except the big blind (player 2)
    for seat in [3, 4, 5, 0]:
        assert s.next_to_act == seat
        s, done, reward, _ = eng.step(s, Action(ActionType.FOLD))
        assert not done
        assert reward is None

    # Small blind folds; big blind should win the pot immediately
    assert s.next_to_act == 1
    s, done, reward, _ = eng.step(s, Action(ActionType.FOLD))
    assert done
    assert reward is not None
    assert len(reward) == 6
    assert sum(reward) == 0

    # Big blind receives the pot, small blind loses their contribution
    expected = [-p.cont for p in s.players]
    winner = 2
    expected[winner] += s.pot
    assert reward == expected


def test_all_players_all_in_preflop_runs_out_board():
    eng = NLHEngine(sb=1, bb=2, start_stack=10, rng=random.Random(0))
    s = eng.reset_hand(button=0)

    actions = [
        (3, Action(ActionType.RAISE_TO, 10)),
        (4, Action(ActionType.CALL)),
        (5, Action(ActionType.CALL)),
        (0, Action(ActionType.CALL)),
        (1, Action(ActionType.CALL)),
        (2, Action(ActionType.CALL)),
    ]

    for seat, action in actions:
        assert s.next_to_act == seat
        s, done, reward, _ = eng.step(s, action)

    assert done
    assert reward is not None
    assert s.round_label == "Showdown"
    assert len(s.board) == 5
    assert s.next_to_act is None
    assert all(p.status == "allin" for p in s.players)
    assert sum(reward) == 0
