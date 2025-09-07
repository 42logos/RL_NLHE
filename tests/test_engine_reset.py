import random
import pytest

from nlhe.core.engine import NLHEngine


@pytest.mark.parametrize(
    "sb, bb, start_stack",
    [
        (1, 2, 100),
        (2, 5, 200),
        (5, 10, 1000),
    ],
)
def test_reset_hand_initializes_state(sb: int, bb: int, start_stack: int):
    eng = NLHEngine(sb=sb, bb=bb, start_stack=start_stack, rng=random.Random(0))
    state = eng.reset_hand(button=0)

    # Pot and betting state
    assert state.pot == sb + bb
    assert state.current_bet == bb
    assert state.next_to_act == (0 + 3) % eng.N

    # Verify hole cards
    seen_cards = set()
    for p in state.players:
        assert p.hole is not None
        assert len(p.hole) == 2
        c1, c2 = p.hole
        assert c1 != c2
        seen_cards.update(p.hole)

    assert len(seen_cards) == eng.N * 2
