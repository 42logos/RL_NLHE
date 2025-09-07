from nlhe.core.state_map import canonical_state, states_equal
from nlhe.core.types import GameState, PlayerState


def _make_state():
    players = [
        PlayerState(hole=(1, 2), stack=100, bet=0, cont=0, status="active", rho=0),
        PlayerState(hole=(3, 4), stack=100, bet=0, cont=0, status="active", rho=0),
    ]
    return GameState(
        button=0,
        round_label="preflop",
        board=[],
        undealt=[5, 6, 7, 8],
        players=players,
        current_bet=0,
        min_raise=2,
        tau=0,
        next_to_act=0,
        step_idx=0,
        pot=0,
        sb=1,
        bb=2,
        actions_log=[],
    )


def test_canonical_state_ignores_non_public_order():
    s1 = _make_state()
    s2 = _make_state()
    s2.undealt = list(reversed(s2.undealt))
    assert canonical_state(s1, include_cards=False) == canonical_state(
        s2, include_cards=False
    )


def test_states_equal_detects_visible_changes():
    s1 = _make_state()
    s2 = _make_state()
    assert states_equal(s1, s2)
    s3 = _make_state()
    s3.pot += 1
    assert not states_equal(s1, s3)
