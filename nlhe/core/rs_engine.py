from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random
from .types import Action as PyAction, ActionType, LegalActionInfo as PyLegalActionInfo

try:
    import importlib.util, importlib.machinery, pathlib
    spec = importlib.util.find_spec("nlhe_engine")
    if spec is None or spec.origin is None:
        raise ImportError("nlhe_engine package not found")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    lib_path = pathlib.Path(spec.origin).with_name(f"nlhe_engine{suffix}")
    spec = importlib.util.spec_from_file_location("nlhe_engine", lib_path)
    _rs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(_rs)  # type: ignore[arg-type]
except Exception as e:  # pragma: no cover - import error path
    raise ImportError(f"Rust backend not found: {e}")

_ACTION_ID = { ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 2, ActionType.RAISE_TO: 3 }

def action_type_id(kind: ActionType) -> int: return _ACTION_ID[kind]
def round_label_id(label: str) -> int: return {"Preflop":0,"Flop":1,"Turn":2,"River":3}.get(label,3)

class NLHEngine:
    def __init__(self, sb: int = 1, bb: int = 2, start_stack: int = 100,
                 num_players: int = 6, rng: Optional[random.Random] = None):
        if num_players != 6:
            raise AssertionError("Engine fixed to 6 players per spec")
        self.sb = int(sb)
        self.bb = int(bb)
        self.start_stack = int(start_stack)
        self.N = int(num_players)
        seed = int(rng.getrandbits(64)) if rng is not None else None
        self._rs = _rs.NLHEngine(
            sb=self.sb,
            bb=self.bb,
            start_stack=self.start_stack,
            num_players=self.N,
            seed=seed,
        )

        # cache Action singletons (no amounts needed for CHECK/CALL/FOLD; RAISE_TO presence only)
        self._act_singleton = {
            ActionType.FOLD: PyAction(ActionType.FOLD),
            ActionType.CHECK: PyAction(ActionType.CHECK),
            ActionType.CALL: PyAction(ActionType.CALL),
            ActionType.RAISE_TO: PyAction(ActionType.RAISE_TO),
        }
        self._mask_cache = {}
        for m in range(16):
            lst = []
            if m & 1:
                lst.append(self._act_singleton[ActionType.FOLD])
            if m & 2:
                lst.append(self._act_singleton[ActionType.CHECK])
            if m & 4:
                lst.append(self._act_singleton[ActionType.CALL])
            if m & 8:
                lst.append(self._act_singleton[ActionType.RAISE_TO])
            self._mask_cache[m] = tuple(lst)
        self._la_reusable = PyLegalActionInfo(
            actions=[],
            min_raise_to=None,
            max_raise_to=None,
            has_raise_right=None,
        )

        self._state: Optional[_rs.GameState] = None

    def reset_hand(self, button: int = 0) -> _rs.GameState:
        if self._state is None:
            self._state = self._rs.reset_hand(int(button))
        else:
            self._rs.reset_hand_apply_py(self._state, int(button))
        return self._state

    def owed(self, s: _rs.GameState, i: int) -> int:
        return max(0, int(s.current_bet) - int(s.players[i].bet))

    def legal_actions(self, s: _rs.GameState) -> PyLegalActionInfo:
        mask, min_to, max_to, has_rr = self._rs.legal_actions_bits_now()
        la = self._la_reusable
        la.actions = self._mask_cache[int(mask)]  # tuple of cached singletons
        la.min_raise_to = None if min_to is None else int(min_to)
        la.max_raise_to = None if max_to is None else int(max_to)
        la.has_raise_right = None if has_rr is None else bool(has_rr)
        
        #return as list
        return PyLegalActionInfo(
            actions=list(la.actions),
            min_raise_to=la.min_raise_to,
            max_raise_to=la.max_raise_to,
            has_raise_right=la.has_raise_right
        )

    def step(
        self, s: _rs.GameState, a: PyAction
    ) -> Tuple[_rs.GameState, bool, Optional[List[int]], Dict[str, Any]]:
        kind = _ACTION_ID[a.kind]
        amt = None if a.amount is None else int(a.amount)
        done, rewards = self._rs.step_apply_py_raw(s, kind, amt)
        return s, bool(done), (None if rewards is None else [int(x) for x in rewards]), {}

    def advance_round_if_needed(self, s: _rs.GameState) -> Tuple[bool, Optional[List[int]]]:
        done, rewards = self._rs.advance_round_if_needed_apply_py(s)
        return bool(done), (None if rewards is None else [int(x) for x in rewards])
