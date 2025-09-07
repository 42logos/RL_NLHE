from nlhe.core.engine import NLHEngine
from nlhe.core.types import GameState, PlayerState


def _make_state(board, holes, conts, statuses):
    players = [
        PlayerState(hole=hole, cont=cont, status=status)
        for hole, cont, status in zip(holes, conts, statuses)
    ]
    return GameState(
        button=0,
        round_label="River",
        board=board,
        undealt=[],
        players=players,
        current_bet=0,
        min_raise=0,
        tau=0,
        next_to_act=None,
        step_idx=0,
        pot=sum(conts),
        sb=1,
        bb=2,
    )


def test_single_main_pot_winner():
    eng = NLHEngine()
    board = [0, 18, 33, 48, 11]
    holes = [
        (12, 25),  # player 0: pair of Aces
        (2, 15),   # player 1: pair of Fours
        None,
        None,
        None,
        None,
    ]
    conts = [50, 50, 0, 0, 0, 0]
    statuses = ["allin", "allin", "folded", "folded", "folded", "folded"]
    s = _make_state(board, holes, conts, statuses)
    rewards = eng._settle_showdown(s)
    assert sum(rewards) == 0
    assert rewards == [50, -50, 0, 0, 0, 0]


def test_split_pot_multiple_winners():
    eng = NLHEngine()
    board = [3, 16, 33, 48, 39]
    holes = [
        (12, 1),    # player 0: pair 5s with Ace kicker
        (25, 2),    # player 1: pair 5s with Ace kicker
        (11, 23),   # player 2: pair 5s with K-Q kickers
        None,
        None,
        None,
    ]
    conts = [30, 30, 30, 0, 0, 0]
    statuses = ["allin", "allin", "allin", "folded", "folded", "folded"]
    s = _make_state(board, holes, conts, statuses)
    rewards = eng._settle_showdown(s)
    assert sum(rewards) == 0
    assert rewards == [15, 15, -30, 0, 0, 0]


def test_side_pot_payouts():
    eng = NLHEngine()
    board = [6, 19, 37, 42, 26]
    holes = [
        (47, 20),  # player 0: T kicker
        (51, 1),   # player 1: A kicker
        (49, 2),   # player 2: Q kicker
        None,
        None,
        None,
    ]
    conts = [100, 50, 100, 0, 0, 0]
    statuses = ["allin", "allin", "allin", "folded", "folded", "folded"]
    s = _make_state(board, holes, conts, statuses)
    rewards = eng._settle_showdown(s)
    assert sum(rewards) == 0
    assert rewards == [-100, 100, 0, 0, 0, 0]
    payouts = [r + c for r, c in zip(rewards, conts)]
    assert payouts[:3] == [0, 150, 100]
