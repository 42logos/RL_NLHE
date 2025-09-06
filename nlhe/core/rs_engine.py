from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random
from .types import Action as PyAction, ActionType, PlayerState as PyPlayerState
from .types import GameState as PyGameState, LegalActionInfo as PyLegalActionInfo

try:
    import nlhe_engine as _rs
except Exception as e:
    raise ImportError(f"Rust backend not found: {e}")

_ACTION_ID = { ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 2, ActionType.RAISE_TO: 3 }
_ID_TO_ACTIONTYPE = { v: k for k, v in _ACTION_ID.items() }

def action_type_id(kind: ActionType) -> int: return _ACTION_ID[kind]
def round_label_id(label: str) -> int: return {"Preflop":0,"Flop":1,"Turn":2,"River":3}.get(label,3)

def _to_rs_action(a: PyAction) -> _rs.Action:
    return _rs.Action(_ACTION_ID[a.kind], None if a.amount is None else int(a.amount))

def _from_rs_legal(li: _rs.LegalActionInfo) -> PyLegalActionInfo:
    # actions list carries kinds only; amounts only for RAISE_TO
    acts = []
    for a in li.actions:
        kind = _ID_TO_ACTIONTYPE[int(a.kind)]
        amt = None if a.amount is None else int(a.amount)
        acts.append(PyAction(kind=kind, amount=amt))
    return PyLegalActionInfo(
        actions=acts,
        min_raise_to=None if li.min_raise_to is None else int(li.min_raise_to),
        max_raise_to=None if li.max_raise_to is None else int(li.max_raise_to),
        has_raise_right=None if li.has_raise_right is None else bool(li.has_raise_right),
    )

def _apply_diff_inplace(s: PyGameState, diff: _rs.StepDiff) -> None:
    # cheap scalars
    s.next_to_act = diff.next_to_act
    s.step_idx    = int(diff.step_idx)
    s.current_bet = int(diff.current_bet)
    s.min_raise   = int(diff.min_raise)
    s.tau         = int(diff.tau)
    s.pot         = int(diff.pot)
    if diff.round_label is not None:
        s.round_label = str(diff.round_label)
    # add dealt cards
    if diff.board_drawn:
        s.board.extend(int(c) for c in diff.board_drawn)
    # append last log if present
    if diff.actions_log_push is not None:
        s.actions_log.append(tuple(int(x) for x in diff.actions_log_push))
    # per-player updates
    for pu in diff.player_updates:
        i = int(pu.idx)
        p = s.players[i]
        p.stack = int(pu.stack)
        p.bet   = int(pu.bet)
        p.cont  = int(pu.cont)
        p.rho   = int(pu.rho)
        if pu.status is not None:
            p.status = str(pu.status)

class NLHEngine:
    def __init__(self, sb: int = 1, bb: int = 2, start_stack: int = 100,
                 num_players: int = 6, rng: Optional[random.Random] = None):
        if num_players != 6:
            raise AssertionError("Engine fixed to 6 players per spec")
        self.sb = int(sb); self.bb = int(bb); self.start_stack = int(start_stack); self.N = int(num_players)
        # derive seed from rng if provided (stable behavior)
        seed = int(rng.getrandbits(64)) if rng is not None else None
        self._rs = _rs.NLHEngine(sb=self.sb, bb=self.bb, start_stack=self.start_stack, num_players=self.N, seed=seed)
        
        # cache Action singletons (no amounts needed for CHECK/CALL/FOLD; RAISE_TO presence only)
        self._act_singleton = {
            ActionType.FOLD:     PyAction(ActionType.FOLD),
            ActionType.CHECK:    PyAction(ActionType.CHECK),
            ActionType.CALL:     PyAction(ActionType.CALL),
            ActionType.RAISE_TO: PyAction(ActionType.RAISE_TO),
        }
        # 16 masks -> tuple of singleton Actions in a stable order
        self._mask_cache = {}
        for m in range(16):
            lst = []
            if m & 1:  lst.append(self._act_singleton[ActionType.FOLD])
            if m & 2:  lst.append(self._act_singleton[ActionType.CHECK])
            if m & 4:  lst.append(self._act_singleton[ActionType.CALL])
            if m & 8:  lst.append(self._act_singleton[ActionType.RAISE_TO])
            self._mask_cache[m] = tuple(lst)   # tuple = immutable, no per-call allocation of Actions
        # reusable LegalActionInfo (we just mutate its fields)
        self._la_reusable = PyLegalActionInfo(actions=[], min_raise_to=None, max_raise_to=None, has_raise_right=None)
        
        # State management for optimization 2 (in-place reset)
        self._state = None
        
        # Cache for in-place reset
        self._state = None

    def reset_hand(self, button: int = 0) -> PyGameState:
        if self._state is None:
            # Create once by asking Rust for a full snapshot, then reuse forever.
            self._state = self._create_initial_state(button)
        else:
            # Fast path: mutate in place and return the same object (API compatible)
            self._rs.reset_hand_apply_py(self._state, int(button))
        return self._state

    def _create_initial_state(self, button: int) -> PyGameState:
        # one-time snapshot to create the Python mirror
        s_rs = self._rs.reset_hand(int(button))
        # convert once (initial)
        players = []
        for pr in s_rs.players:
            players.append(PyPlayerState(
                hole=None if pr.hole is None else (int(pr.hole[0]), int(pr.hole[1])),
                stack=int(pr.stack), bet=int(pr.bet), cont=int(pr.cont),
                status=str(pr.status), rho=int(pr.rho)
            ))
        s = PyGameState(
            button=int(s_rs.button),
            round_label=str(s_rs.round_label),
            board=[int(x) for x in s_rs.board],
            undealt=[int(x) for x in s_rs.undealt],
            players=players,
            current_bet=int(s_rs.current_bet),
            min_raise=int(s_rs.min_raise),
            tau=int(s_rs.tau),
            next_to_act=None if s_rs.next_to_act is None else int(s_rs.next_to_act),
            step_idx=int(s_rs.step_idx),
            pot=int(s_rs.pot),
            sb=self.sb, bb=self.bb,
            actions_log=[(int(i),int(a),int(v),int(r)) for (i,a,v,r) in s_rs.actions_log],
        )
        return s

    # Cheap helper stays in Python
    def owed(self, s: PyGameState, i: int) -> int:
        return max(0, int(s.current_bet) - int(s.players[i].bet))

    def legal_actions(self, s: PyGameState) -> PyLegalActionInfo:
        mask, min_to, max_to, has_rr = self._rs.legal_actions_bits_now()
        la = self._la_reusable
        la.actions = self._mask_cache[int(mask)]  # tuple of cached singletons
        la.min_raise_to = None if min_to is None else int(min_to)
        la.max_raise_to = None if max_to is None else int(max_to)
        la.has_raise_right = None if has_rr is None else bool(has_rr)
        return la

    def step(self, s: PyGameState, a: PyAction) -> Tuple[PyGameState, bool, Optional[List[int]], Dict[str, Any]]:
        # map Action → two scalars (no PyO3 Action at all)
        kind = _ACTION_ID[a.kind]
        amt = None if a.amount is None else int(a.amount)
        done, rewards = self._rs.step_apply_py_raw(s, kind, amt)
        return s, bool(done), (None if rewards is None else [int(x) for x in rewards]), {}

    def advance_round_if_needed(self, s: PyGameState) -> Tuple[bool, Optional[List[int]]]:
        done, rewards = self._rs.advance_round_if_needed_apply_py(s)
        return bool(done), (None if rewards is None else [int(x) for x in rewards])
