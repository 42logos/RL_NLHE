import random
import sys
import types

import pytest




from nlhe.core.engine import NLHEngine as PyEngine
from nlhe.core.state_map import canonical_state

_stub = types.ModuleType("nlhe_engine")
sys.modules.setdefault("nlhe_engine", _stub)
# Remove stub so importing the Rust engine will fail and the test will skip.
sys.modules.pop("nlhe_engine", None)

try:
    from nlhe.core.rs_engine import NLHEngine as RsEngine
    print("✓ Rust engine available for testing")
except ImportError as e:
    print(f"✗ Rust engine not available: {e}")
    sys.exit(1)


from nlhe.core.types import Action, ActionType


def _sample_action(rng: random.Random, info):
    """Sample a legal action using ``info`` describing legal moves."""
    act = rng.choice(info.actions)
    if act.kind is ActionType.RAISE_TO:
        assert info.max_raise_to is not None
        min_to = info.min_raise_to if info.min_raise_to is not None else info.max_raise_to
        max_to = info.max_raise_to
        if min_to >= max_to:
            target = max_to
        else:
            target = rng.randint(min_to, max_to)
        return Action(ActionType.RAISE_TO, target)
    return act


@pytest.mark.parametrize("seed", [0, 1])
def test_python_vs_rust_engine_equivalence(seed: int):
    py_rng = random.Random(seed)
    rs_rng = random.Random(seed)
    action_rng = random.Random(seed)

    py = PyEngine(rng=py_rng)
    rs = RsEngine(rng=rs_rng)

    s_py = py.reset_hand(button=0)
    s_rs = rs.reset_hand(button=0)
    assert canonical_state(s_py, include_cards=False) == canonical_state(
        s_rs, include_cards=False
    )

    for _ in range(200):
        li_py = py.legal_actions(s_py)
        li_rs = rs.legal_actions(s_rs)
        assert li_py == li_rs
        if not li_py.actions:
            break
        a = _sample_action(action_rng, li_py)
        s_py, done_py, rew_py, _ = py.step(s_py, a)
        s_rs, done_rs, rew_rs, _ = rs.step(s_rs, a)
        assert done_py == done_rs
        if done_py:
            assert rew_py is not None and rew_rs is not None
            assert len(rew_py) == len(rew_rs)
            assert sum(rew_py) == 0 and sum(rew_rs) == 0
        else:
            assert rew_py is None and rew_rs is None
        assert canonical_state(s_py, include_cards=False) == canonical_state(
            s_rs, include_cards=False
        )
        assert s_py.pot == sum(p.cont for p in s_py.players)
        assert s_py.current_bet == max(p.bet for p in s_py.players)
        if done_py:
            break

@pytest.mark.parametrize("seed", [i for i in range(23)])
def test_rs_engine_complete_parity(seed: int):
    """Test that the Rust engine is fully equivalent to the Python engine."""
    rng = random.Random(seed)
    py=PyEngine(rng=rng)
    rs=RsEngine(rng=rng)
    
    py_state = py.reset_hand(button=0)
    rs_state = rs.reset_hand(button=0)
    
    unsucc=[]
    for i, pyP in enumerate(py_state.players):
        try:
            rs._test_set_player_hole(i, (pyP.hole[0], pyP.hole[1]))
        except Exception as e:
            print(f"Error setting player {i} hole cards in rust engine: {e}")
            unsucc.append(i)
    
    rerun_count=0
    while len(unsucc)>0:
        print(f"Retrying {len(unsucc)} failed players...")
        print(f"Current unsucc players: {unsucc}")
        i=unsucc.pop(0)
        pyP = py_state.players[i]
        try:
            rs._test_set_player_hole(i, (pyP.hole[0], pyP.hole[1]))
        except Exception as e:
            print(f"Retry Error setting player {i} hole cards in Rust engine: {e}")
            unsucc.append(i)
        if len(unsucc)==2:
            rerun_count+=1
            if rerun_count>5:
                Warning.warn(f"deadlock detected setting player holes, aborting retries, posibly situation like  player 1 has card A, player 2 has card B, and A and B are the same card")
                break
    
    # handling deadlock
    for i in unsucc:
        pyP = py_state.players[i]
        print(f"first dealing an undealt card to player {i}")
        card1 = py_state.undealt.pop(0) 
        while card1 in [c for p in py_state.players for c in p.hole] or card1 in py_state.board:
            py_state.undealt.append(card1)
            card1 = py_state.undealt.pop(0)
        card2 = py_state.undealt.pop(0)
        while card2 in [c for p in py_state.players for c in p.hole] or card2 in py_state.board or card2==card1:
            py_state.undealt.append(card2)
            card2 = py_state.undealt.pop(0)
        pyP.hole = (card1, card2)
        print(f"player {i} assigned cards {card1}, {card2}")
        try:
            rs._test_set_player_hole(i, (pyP.hole[0], pyP.hole[1]))
        except Exception as e:
            print(f"Error setting player {i} hole cards in rust engine: {e}")
            unsucc.append(i)
    assert len(unsucc)==0, f"Failed to set hole cards for players {unsucc} in rust engine"
    rs._test_set_undealt(list(py_state.undealt))

    print(f"py state players: {[ (p.stack, p.hole) for p in py_state.players ]}")
    print(f"rs state players: {[ (p.stack, p.hole) for p in rs_state.players ]}")  
    for attr in ['board', 'undealt', 'button', 'round_label', 'current_bet', 'min_raise', 'tau', 'next_to_act', 'step_idx', 'pot', 'sb', 'bb']:
        py_val = getattr(py_state, attr)
        rs_val = getattr(rs_state, attr)
        assert py_val == rs_val, f"Mismatch in attribute {attr}: py={py_val}, rs={rs_val}"
    assert canonical_state(py_state) == canonical_state(rs_state)
    
    for _ in range(410):
        li_py = py.legal_actions(py_state)
        li_rs = rs.legal_actions(rs_state)
        assert li_py == li_rs
        if not li_py.actions:
            break
        a = py.rng.choice(li_py.actions)
        if a.kind is ActionType.RAISE_TO:
            assert li_py.max_raise_to is not None
            min_to = li_py.min_raise_to if li_py.min_raise_to is not None else li_py.max_raise_to
            max_to = li_py.max_raise_to
            if min_to >= max_to:
                target = max_to
            else:
                target = py.rng.randint(min_to, max_to)
            a = Action(ActionType.RAISE_TO, target)
        py_state, done_py, rew_py, _ = py.step(py_state, a)
        rs_state, done_rs, rew_rs, _ = rs.step(rs_state, a)
        rs._test_set_board(list(py_state.board))
        rs._test_set_undealt(list(py_state.undealt))
        assert canonical_state(py_state, include_cards=True) == canonical_state(rs_state, include_cards=True)
        assert done_py == done_rs
        if done_py:
            assert rew_py is not None and rew_rs is not None
            assert len(rew_py) == len(rew_rs)
            assert sum(rew_py) == 0 and sum(rew_rs) == 0
            print(f"Final state py: {canonical_state(py_state)}")
            print(f"Final state rs: {canonical_state(rs_state)}")
            assert rew_py == rew_rs
        else:
            assert rew_py is None and rew_rs is None
        for attr in ['board', 'undealt', 'button', 'round_label', 'current_bet', 'min_raise', 'tau', 'next_to_act', 'step_idx', 'pot', 'sb', 'bb']:
            py_val = getattr(py_state, attr)
            rs_val = getattr(rs_state, attr)
            assert py_val == rs_val, f"Mismatch in attribute {attr}: py={py_val}, rs={rs_val}"
        assert canonical_state(py_state) == canonical_state(rs_state)
        assert py_state.pot == sum(p.cont for p in py_state.players)
        assert py_state.current_bet == max(p.bet for p in py_state.players)
        if done_py:
            break