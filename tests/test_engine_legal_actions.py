import pytest

from nlhe.core.engine import NLHEngine
from nlhe.core.types import Action, ActionType


def test_owing_chips_can_raise_open_rights():
    eng = NLHEngine()
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


def test_owing_chips_cannot_raise_short_stack():
    eng = NLHEngine()
    s = eng.reset_hand(button=0)
    i = s.next_to_act
    p = s.players[i]
    p.stack = eng.owed(s, i)  # only enough to call
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
