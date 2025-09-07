import copy
import random
import sys
import types

import pytest

# Provide a minimal stub for the optional compiled evaluator so the engine
# can be imported without the Rust extension being available.
_stub = types.ModuleType("nlhe_engine")
_stub.best5_rank_from_7_py = lambda cards: (0, [0])
sys.modules.setdefault("nlhe_engine", _stub)

from nlhe.core.engine import NLHEngine
from nlhe.core.types import Action, ActionType

# Remove stub so other tests that expect the Rust backend to be missing will
# see an ImportError and skip appropriately.
sys.modules.pop("nlhe_engine", None)


def test_fold_step():
    eng = NLHEngine(rng=random.Random(0))
    s = eng.reset_hand(button=0)

    # Player 3 is first to act and owes the big blind
    p = s.players[3]
    assert p.status == "active"
    s, done, reward, _ = eng.step(s, Action(ActionType.FOLD))
    assert not done and reward is None

    p = s.players[3]
    assert p.status == "folded"
    assert p.stack == 100
    assert p.bet == 0
    assert p.cont == 0
    assert s.pot == 3
    assert s.current_bet == 2
    assert s.next_to_act == 4
    assert s.actions_log[-1] == (3, 0, 0, 0)


def test_check_step():
    eng = NLHEngine(rng=random.Random(0))
    s = eng.reset_hand(button=0)

    # Make the big blind check
    s.next_to_act = 2
    s, done, reward, _ = eng.step(s, Action(ActionType.CHECK))
    assert not done and reward is None

    p = s.players[2]
    assert p.status == "active"
    assert p.stack == 98
    assert p.bet == 2
    assert p.cont == 2
    assert s.pot == 3
    assert s.current_bet == 2
    assert s.next_to_act == 3
    assert s.actions_log[-1] == (2, 1, 0, 0)


def test_call_step():
    eng = NLHEngine(rng=random.Random(0))
    s = eng.reset_hand(button=0)

    s, done, reward, _ = eng.step(s, Action(ActionType.CALL))
    assert not done and reward is None

    p = s.players[3]
    assert p.status == "active"
    assert p.stack == 98
    assert p.bet == 2
    assert p.cont == 2
    assert s.pot == 5
    assert s.current_bet == 2
    assert s.next_to_act == 4
    assert s.actions_log[-1] == (3, 2, 0, 0)


def test_raise_to_step():
    eng = NLHEngine(rng=random.Random(0))
    s = eng.reset_hand(button=0)

    s, done, reward, _ = eng.step(s, Action(ActionType.RAISE_TO, 6))
    assert not done and reward is None

    p = s.players[3]
    assert p.status == "active"
    assert p.stack == 94
    assert p.bet == 6
    assert p.cont == 6
    assert s.pot == 9
    assert s.current_bet == 6
    assert s.next_to_act == 4
    assert s.actions_log[-1] == (3, 3, 6, 0)


def test_raise_rights_close_after_full_raise():
    eng = NLHEngine(rng=random.Random(0))
    s = eng.reset_hand(button=0)

    # Only players 3,4,5 remain active
    for idx in range(3):
        s.players[idx].status = "folded"
    s.next_to_act = 3

    # Player 3 makes a full raise to 6
    s, _, _, _ = eng.step(s, Action(ActionType.RAISE_TO, 6))
    # Player 4 calls
    s, _, _, _ = eng.step(s, Action(ActionType.CALL))
    # Player 5 is short and can only raise all-in to 9 (< required full raise)
    s.players[5].stack = 9
    s, _, _, _ = eng.step(s, Action(ActionType.RAISE_TO, 9))

    # Back to player 3 who has no raise rights
    li = eng.legal_actions(s)
    assert li.has_raise_right is False

    s_copy = copy.deepcopy(s)
    with pytest.raises(AssertionError):
        eng.step(s_copy, Action(ActionType.RAISE_TO, li.min_raise_to))
