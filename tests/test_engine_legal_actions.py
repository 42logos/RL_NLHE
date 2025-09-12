import pytest

from nlhe.core.engine import NLHEngine
from nlhe.core.rs_engine import NLHEngine as Rs_nlheEngine
from nlhe.core.types import Action, ActionType

@pytest.mark.parametrize("engine_class", [NLHEngine, Rs_nlheEngine])
def test_owing_chips(engine_class):
    eng = engine_class()
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    assert eng.owed(s, i) == eng.bb - p.bet  # big blind to call
    # Simulate some betting
    p.bet += 3
    p.stack -= 3
    p.cont += 3
    s.pot += 3
    s.current_bet = p.bet
    s.min_raise = 2 * eng.bb
    i = s.next_to_act
    p = s.players[i]
    assert eng.owed(s, i) == s.current_bet - p.bet  # amount to call

@pytest.mark.parametrize("engine_class", [NLHEngine, Rs_nlheEngine])
def test_owing_chips_can_raise_open_rights(engine_class):
    eng = engine_class()
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    info = eng.legal_actions(s)
    assert info.actions == [
        Action(ActionType.FOLD),
        Action(ActionType.CALL),
        Action(ActionType.RAISE_TO),
    ]
    assert info.min_raise_to == s.current_bet + s.min_raise
    assert info.max_raise_to == p.bet + p.stack
    assert info.has_raise_right is True

@pytest.mark.parametrize("engine_class", [NLHEngine, Rs_nlheEngine])
def test_owing_chips_cannot_raise_short_stack(engine_class):
    eng = engine_class(start_stack=2)
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    info = eng.legal_actions(s)
    assert info.actions == [
        Action(ActionType.FOLD),
        Action(ActionType.CALL),
    ]
    assert info.min_raise_to is None
    assert info.max_raise_to is None
    assert info.has_raise_right is None


def test_no_outstanding_bet_check_available():
    eng = NLHEngine()
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    owe = eng.owed(s, i)
    p.bet += owe
    p.stack -= owe
    p.cont += owe
    s.pot += owe
    info = eng.legal_actions(s)
    assert info.actions == [
        Action(ActionType.CHECK),
        Action(ActionType.RAISE_TO),
    ]
    assert info.min_raise_to == s.current_bet + s.min_raise
    assert info.max_raise_to == p.bet + p.stack
    assert info.has_raise_right is True


def test_non_active_player_has_no_actions():
    eng = NLHEngine()
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    p.status = "allin"
    p.stack = 0
    info = eng.legal_actions(s)
    assert info.actions == []
    assert info.min_raise_to is None
    assert info.max_raise_to is None
    assert info.has_raise_right is None


def test_closed_raise_rights_flagged():
    eng = NLHEngine()
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    p.rho = s.tau  # close raise rights
    info = eng.legal_actions(s)
    assert info.actions == [
        Action(ActionType.FOLD),
        Action(ActionType.CALL),
        Action(ActionType.RAISE_TO),
    ]
    assert info.min_raise_to == s.current_bet + s.min_raise
    assert info.max_raise_to == p.bet + p.stack
    assert info.has_raise_right is False
