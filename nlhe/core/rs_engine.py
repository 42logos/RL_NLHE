from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random

# --- Keep your public imports/types exactly the same ---
from .types import Action as PyAction, ActionType, PlayerState as PyPlayerState
from .types import GameState as PyGameState, LegalActionInfo as PyLegalActionInfo

# Keep your historical IDs so logs/agents remain stable
_ACTION_ID = { ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 2, ActionType.RAISE_TO: 3 }
_ID_TO_ACTIONTYPE = { v: k for k, v in _ACTION_ID.items() }

try:
    # Rust backend (from the crate we built)
    import nlhe_engine as _rs
    _HAVE_RS = True
except Exception as e:
    _HAVE_RS = False
    _RS_IMPORT_ERR = e


def action_type_id(kind: ActionType) -> int:
    return _ACTION_ID[kind]

def round_label_id(label: str) -> int:
    return {"Preflop": 0, "Flop": 1, "Turn": 2, "River": 3}.get(label, 3)


# ----------------------------
# Converters: Python <-> Rust
# ----------------------------
def _to_rs_action(a: PyAction) -> _rs.Action:
    kid = _ACTION_ID[a.kind]
    amt = None if a.amount is None else int(a.amount)
    return _rs.Action(kid, amt)

def _from_rs_action(a_rs: _rs.Action) -> PyAction:
    kind = _ID_TO_ACTIONTYPE[int(a_rs.kind)]
    return PyAction(kind=kind, amount=(None if a_rs.amount is None else int(a_rs.amount)))

def _to_rs_player(p: PyPlayerState) -> _rs.PlayerState:
    hole = None if p.hole is None else (int(p.hole[0]), int(p.hole[1]))
    return _rs.PlayerState(
        hole=hole,
        stack=int(p.stack),
        bet=int(p.bet),
        cont=int(p.cont),
        status=str(p.status),
        rho=int(p.rho),
    )

def _from_rs_player(p_rs: _rs.PlayerState) -> PyPlayerState:
    return PyPlayerState(
        hole=None if p_rs.hole is None else (int(p_rs.hole[0]), int(p_rs.hole[1])),
        stack=int(p_rs.stack),
        bet=int(p_rs.bet),
        cont=int(p_rs.cont),
        status=str(p_rs.status),
        rho=int(p_rs.rho),
    )

def _to_rs_state(s: PyGameState) -> _rs.GameState:
    players_rs = [_to_rs_player(p) for p in s.players]
    return _rs.GameState(
        button=int(s.button),
        round_label=str(s.round_label),
        board=[int(x) for x in s.board],
        undealt=[int(x) for x in s.undealt],
        players=players_rs,
        current_bet=int(s.current_bet),
        min_raise=int(s.min_raise),
        tau=int(s.tau),
        next_to_act=None if s.next_to_act is None else int(s.next_to_act),
        step_idx=int(s.step_idx),
        pot=int(s.pot),
        sb=int(s.sb),
        bb=int(s.bb),
    )

def _from_rs_state(s_rs: _rs.GameState) -> PyGameState:
    return PyGameState(
        button=int(s_rs.button),
        round_label=str(s_rs.round_label),
        board=[int(x) for x in s_rs.board],
        undealt=[int(x) for x in s_rs.undealt],
        players=[_from_rs_player(p) for p in s_rs.players],
        current_bet=int(s_rs.current_bet),
        min_raise=int(s_rs.min_raise),
        tau=int(s_rs.tau),
        next_to_act=None if s_rs.next_to_act is None else int(s_rs.next_to_act),
        step_idx=int(s_rs.step_idx),
        pot=int(s_rs.pot),
        sb=int(s_rs.sb),
        bb=int(s_rs.bb),
        actions_log=[(int(i), int(a), int(v), int(r)) for (i, a, v, r) in s_rs.actions_log],
    )

def _from_rs_legal(li: _rs.LegalActionInfo) -> PyLegalActionInfo:
    acts = [_from_rs_action(a) for a in li.actions]
    return PyLegalActionInfo(
        actions=acts,
        min_raise_to=None if li.min_raise_to is None else int(li.min_raise_to),
        max_raise_to=None if li.max_raise_to is None else int(li.max_raise_to),
        has_raise_right=None if li.has_raise_right is None else bool(li.has_raise_right),
    )


# ----------------------------
# Public Engine (Shim over Rust)
# ----------------------------
class NLHEngine:
    """
    Drop-in class compatible with your existing users:

        eng = NLHEngine(sb=1, bb=2, start_stack=100, rng=random.Random(42))
        s = eng.reset_hand(button=0)
        info = eng.legal_actions(s)
        s, done, rewards, info = eng.step(s, Action(ActionType.CALL))

    Internally calls the Rust engine for speed, converting to/from your dataclasses.
    """

    def __init__(self, sb: int = 1, bb: int = 2, start_stack: int = 100,
                 num_players: int = 6, rng: Optional[random.Random] = None):
        if not _HAVE_RS:
            raise ImportError(
                f"Rust backend (nlhe_engine) not found: {_RS_IMPORT_ERR}\n"
                "Build it with `maturin develop --release` in nlhe/rs_engine."
            )
        if num_players != 6:
            raise AssertionError("Engine fixed to 6 players per spec")
        self.sb = int(sb)
        self.bb = int(bb)
        self.start_stack = int(start_stack)
        self.N = int(num_players)
        # Derive a reproducible seed when rng is provided
        if rng is not None:
            try:
                seed = int(rng.getrandbits(64))
            except Exception:
                # Fallback if custom Random-like object
                seed = int(random.Random().getrandbits(64))
        else:
            seed = None
        self._rs = _rs.NLHEngine(sb=self.sb, bb=self.bb, start_stack=self.start_stack,
                                 num_players=self.N, seed=seed)

    # ----- lifecycle -----
    def reset_hand(self, button: int = 0) -> PyGameState:
        s_rs = self._rs.reset_hand(int(button))
        return _from_rs_state(s_rs)

    # ----- helpers -----
    def owed(self, s: PyGameState, i: int) -> int:
        # Cheap to compute directly without round-trip:
        return max(0, int(s.current_bet) - int(s.players[i].bet))

    def legal_actions(self, s: PyGameState) -> PyLegalActionInfo:
        s_rs = _to_rs_state(s)
        li_rs = self._rs.legal_actions(s_rs)
        return _from_rs_legal(li_rs)

    # ----- step -----
    def step(self, s: PyGameState, a: PyAction) -> Tuple[PyGameState, bool, Optional[List[int]], Dict[str, Any]]:
        s_rs = _to_rs_state(s)
        a_rs = _to_rs_action(a)
        s_rs2, done, rewards, _info = self._rs.step(s_rs, a_rs)
        s2 = _from_rs_state(s_rs2)
        # info dict kept empty for parity with your code
        return s2, bool(done), (None if rewards is None else [int(x) for x in rewards]), {}

    # ----- public advance / showdown -----
    def advance_round_if_needed(self, s: PyGameState) -> Tuple[bool, Optional[List[int]]]:
        s_rs = _to_rs_state(s)
        done, rewards = self._rs.advance_round_if_needed(s_rs)
        return bool(done), (None if rewards is None else [int(x) for x in rewards])
