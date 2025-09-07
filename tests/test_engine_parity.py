import random

import pytest

from nlhe.core.engine import NLHEngine as PyEngine
from nlhe.core.state_map import canonical_state

# Attempt to import the Rust engine; skip the test if the compiled backend
# is unavailable in the current environment.
rs_mod = pytest.importorskip("nlhe.core.rs_engine")
try:
    import nlhe_engine as _nlhe_mod  # type: ignore
except Exception:  # pragma: no cover - handled by skip below
    _nlhe_mod = None  # pragma: no cover

if _nlhe_mod is None or not hasattr(_nlhe_mod, "NLHEngine"):
    pytest.skip("Rust backend not available", allow_module_level=True)

RsEngine = rs_mod.NLHEngine
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
