from __future__ import annotations
import types
from typing import Any, Dict, List, Optional, Tuple
import random
from .types import Action as PyAction, ActionType, LegalActionInfo as PyLegalActionInfo

import sys
_stub = types.ModuleType("nlhe_engine")
sys.modules.setdefault("nlhe_engine", _stub)
# Remove stub so importing the Rust engine will fail and the test will skip.
sys.modules.pop("nlhe_engine", None)

try:
    # 顶层扩展形态
    import nlhe_engine as _rs
except Exception:
    # 包+子模块形态
    from nlhe_engine import nlhe_engine as _rs


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

        self._state = None
        self._empty_info: Dict[str, Any] = {}

    def reset_hand(self, button: int = 0):
        if self._state is None:
            self._state = self._rs.reset_hand(int(button))
        else:
            self._rs.reset_hand_apply_py(self._state, int(button))
        return self._state

    # Cheap helper stays in Python
    def owed(self, s, i: int) -> int:
        return max(0, int(s.current_bet) - int(s.players[i].bet))

    def legal_actions(self, s) -> PyLegalActionInfo:
        mask, min_to, max_to, has_rr = self._rs.legal_actions_bits_now()
        la = self._la_reusable
        acts = self._mask_cache[int(mask)]  # tuple of cached singletons
        la.actions.clear()
        la.actions.extend(acts)
        la.min_raise_to = None if min_to is None else int(min_to)
        la.max_raise_to = None if max_to is None else int(max_to)
        la.has_raise_right = None if has_rr is None else bool(has_rr)
        return la

    def step(self, s, a: PyAction) -> Tuple[Any, bool, Optional[List[int]], Dict[str, Any]]:
        # map Action → two scalars (no PyO3 Action at all)
        kind = _ACTION_ID[a.kind]
        amt = None if a.amount is None else int(a.amount)
        done, rewards = self._rs.step_apply_py_raw(s, kind, amt)
        return s, bool(done), (None if rewards is None else [int(x) for x in rewards]), self._empty_info

    def advance_round_if_needed(self, s) -> Tuple[bool, Optional[List[int]]]:
        done, rewards = self._rs.advance_round_if_needed_apply_py(s)
        return bool(done), (None if rewards is None else [int(x) for x in rewards])

    # ==============================
    # INTERNAL TEST-ONLY STATE SETTER METHODS
    # ⚠️  WARNING: DO NOT USE THESE METHODS IN PRODUCTION/TRAINING CODE!
    # ⚠️  These methods are for testing purposes only and will significantly
    # ⚠️  slow down engine performance due to validation overhead.
    # ⚠️  Use only for creating specific test scenarios.
    # ==============================
    
    def _test_set_pot(self, new_pot: int) -> None:
        """
        🧪 TEST-ONLY: Set the pot value directly.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            new_pot: New pot value (must be non-negative)
        """
        import warnings
        warnings.warn(
            "Using _test_set_pot() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_pot(int(new_pot))
    
    def _test_set_current_bet(self, new_current_bet: int) -> None:
        """
        🧪 TEST-ONLY: Set the current bet value.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_current_bet() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_current_bet(int(new_current_bet))
    
    def _test_set_min_raise(self, new_min_raise: int) -> None:
        """
        🧪 TEST-ONLY: Set the minimum raise amount.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_min_raise() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_min_raise(int(new_min_raise))
    
    def _test_set_tau(self, new_tau: int) -> None:
        """
        🧪 TEST-ONLY: Set the tau value (step index for raise rights).
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_tau() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_tau(int(new_tau))
    
    def _test_set_step_idx(self, new_step_idx: int) -> None:
        """
        🧪 TEST-ONLY: Set the step index.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_step_idx() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_step_idx(int(new_step_idx))
    
    def _test_set_player_stack(self, player_idx: int, new_stack: int) -> None:
        """
        🧪 TEST-ONLY: Set a player's stack.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_player_stack() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_player_stack(int(player_idx), int(new_stack))
    
    def _test_set_player_bet(self, player_idx: int, new_bet: int) -> None:
        """
        🧪 TEST-ONLY: Set a player's bet amount.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_player_bet() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_player_bet(int(player_idx), int(new_bet))
    
    def _test_set_player_cont(self, player_idx: int, new_cont: int) -> None:
        """
        🧪 TEST-ONLY: Set a player's contribution.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_player_cont() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_player_cont(int(player_idx), int(new_cont))
    
    def _test_set_player_rho(self, player_idx: int, new_rho: int) -> None:
        """
        🧪 TEST-ONLY: Set a player's rho value.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_player_rho() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        self._rs.set_player_rho(int(player_idx), int(new_rho))
    
    def _test_set_player_status(self, player_idx: int, new_status: str) -> None:
        """
        🧪 TEST-ONLY: Set a player's status.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        """
        import warnings
        warnings.warn(
            "Using _test_set_player_status() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        if new_status not in ["active", "folded", "allin"]:
            raise ValueError("status must be 'active', 'folded', or 'allin'")
        self._rs.set_player_status(int(player_idx), str(new_status))
    
    def _test_set_state_batch(self, updates: Dict[str, int]) -> None:
        """
        🧪 TEST-ONLY: Set multiple state variables at once.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            updates: Dictionary mapping variable names to new values.
                    Supported keys: 'pot', 'current_bet', 'min_raise', 'tau', 'step_idx'
        """
        import warnings
        warnings.warn(
            "Using _test_set_state_batch() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        rust_updates = {str(k): int(v) for k, v in updates.items()}
        self._rs.set_state_batch(rust_updates)
    
    def _test_validate_and_fix_state(self, fix_issues: bool = False) -> List[str]:
        """
        🧪 TEST-ONLY: Validate current state consistency.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            fix_issues: If True, automatically fix detected inconsistencies.
        
        Returns:
            List of warning messages about inconsistencies found.
        """
        import warnings
        warnings.warn(
            "Using _test_validate_and_fix_state() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        return self._rs.validate_and_fix_state(bool(fix_issues))

    # ==============================
    # INTERNAL TEST-ONLY CARD SETTER METHODS  
    # ⚠️  WARNING: DO NOT USE THESE METHODS IN PRODUCTION/TRAINING CODE!
    # ⚠️  These methods are for testing purposes only and will significantly
    # ⚠️  slow down engine performance due to validation overhead.
    # ⚠️  Use only for creating specific test scenarios.
    # ==============================
    
    def _test_set_board(self, board_cards: List[int]) -> None:
        """
        🧪 TEST-ONLY: Set the board cards.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            board_cards: List of card values (0-51). Can be empty, or have 3, 4, or 5 cards.
        """
        import warnings
        warnings.warn(
            "Using _test_set_board() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        
        # Validate input
        if not isinstance(board_cards, list):
            raise TypeError("board_cards must be a list")
        
        for card in board_cards:
            if not isinstance(card, int) or card < 0 or card > 51:
                raise ValueError(f"invalid card value: {card} (must be 0-51)")
        
        # Convert to u8 list for Rust
        rust_cards = [int(card) for card in board_cards]
        self._rs.set_board(rust_cards)
    
    def _test_set_player_hole(self, player_idx: int, hole_cards: Optional[Tuple[int, int]]) -> None:
        """
        🧪 TEST-ONLY: Set a player's hole cards.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            player_idx: Player index (0-5)
            hole_cards: Tuple of two card values (0-51), or None to remove hole cards
        """
        import warnings
        warnings.warn(
            "Using _test_set_player_hole() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        
        if not isinstance(player_idx, int) or player_idx < 0:
            raise ValueError("player_idx must be a non-negative integer")
        
        if hole_cards is not None:
            if not isinstance(hole_cards, tuple) or len(hole_cards) != 2:
                raise TypeError("hole_cards must be a tuple of two integers or None")
            
            c1, c2 = hole_cards
            if not isinstance(c1, int) or not isinstance(c2, int):
                raise TypeError("hole cards must be integers")
            
            if c1 < 0 or c1 > 51 or c2 < 0 or c2 > 51:
                raise ValueError("hole cards must be in range 0-51")
            
            if c1 == c2:
                raise ValueError("hole cards cannot be identical")
            
            rust_hole = (int(c1), int(c2))
        else:
            rust_hole = None
        
        self._rs.set_player_hole(int(player_idx), rust_hole)
    
    def _test_set_undealt(self, undealt_cards: List[int]) -> None:
        """
        🧪 TEST-ONLY: Set the undealt deck directly.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            undealt_cards: List of card values (0-51) representing the remaining deck
        """
        import warnings
        warnings.warn(
            "Using _test_set_undealt() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        
        if not isinstance(undealt_cards, list):
            raise TypeError("undealt_cards must be a list")
        
        for card in undealt_cards:
            if not isinstance(card, int) or card < 0 or card > 51:
                raise ValueError(f"invalid card value: {card} (must be 0-51)")
        
        rust_cards = [int(card) for card in undealt_cards]
        self._rs.set_undealt(rust_cards)
    
    def _test_set_card_scenario(self, 
                         board: List[int], 
                         hole_cards: List[Optional[Tuple[int, int]]], 
                         undealt: Optional[List[int]] = None) -> None:
        """
        🧪 TEST-ONLY: Set up a complete card scenario.
        
        ⚠️  WARNING: This method is for testing purposes only!
        ⚠️  DO NOT use in production/training code - will slow down performance.
        
        Args:
            board: List of board card values (0-51)
            hole_cards: List of hole card tuples for each player (or None for no cards)
            undealt: Optional list of remaining deck cards (auto-generated if not provided)
        """
        import warnings
        warnings.warn(
            "Using _test_set_card_scenario() - This is a test-only method that impacts performance. "
            "Do not use in production or training code!",
            UserWarning,
            stacklevel=2
        )
        
        # Validate board
        if not isinstance(board, list):
            raise TypeError("board must be a list")
        
        for card in board:
            if not isinstance(card, int) or card < 0 or card > 51:
                raise ValueError(f"invalid board card: {card}")
        
        # Validate hole_cards
        if not isinstance(hole_cards, list):
            raise TypeError("hole_cards must be a list")
        
        rust_hole_cards = []
        for i, hole in enumerate(hole_cards):
            if hole is None:
                rust_hole_cards.append(None)
            else:
                if not isinstance(hole, tuple) or len(hole) != 2:
                    raise TypeError(f"hole_cards[{i}] must be a tuple of two integers or None")
                
                c1, c2 = hole
                if not isinstance(c1, int) or not isinstance(c2, int):
                    raise TypeError(f"hole_cards[{i}] must contain integers")
                
                if c1 < 0 or c1 > 51 or c2 < 0 or c2 > 51:
                    raise ValueError(f"hole_cards[{i}] must be in range 0-51")
                
                rust_hole_cards.append((int(c1), int(c2)))
        
        # Validate undealt if provided
        rust_undealt = None
        if undealt is not None:
            if not isinstance(undealt, list):
                raise TypeError("undealt must be a list or None")
            
            for card in undealt:
                if not isinstance(card, int) or card < 0 or card > 51:
                    raise ValueError(f"invalid undealt card: {card}")
            
            rust_undealt = [int(card) for card in undealt]
        
        # Convert board to rust format
        rust_board = [int(card) for card in board]
        
        self._rs.set_card_scenario(rust_board, rust_hole_cards, rust_undealt)

    # ==============================
    # DEPRECATED: Backward compatibility methods 
    # These methods are deprecated and will issue warnings
    # ==============================
    
    def set_pot(self, new_pot: int) -> None:
        """DEPRECATED: Use _test_set_pot() instead. This method will be removed."""
        import warnings
        warnings.warn(
            "set_pot() is deprecated. Use _test_set_pot() for testing only. "
            "These methods should not be used in production/training code!",
            DeprecationWarning,
            stacklevel=2
        )
        self._test_set_pot(new_pot)
    
    def set_current_bet(self, new_current_bet: int) -> None:
        """DEPRECATED: Use _test_set_current_bet() instead. This method will be removed."""
        import warnings
        warnings.warn(
            "set_current_bet() is deprecated. Use _test_set_current_bet() for testing only. "
            "These methods should not be used in production/training code!",
            DeprecationWarning,
            stacklevel=2
        )
        self._test_set_current_bet(new_current_bet)
    
    def set_min_raise(self, new_min_raise: int) -> None:
        """DEPRECATED: Use _test_set_min_raise() instead. This method will be removed."""
        import warnings
        warnings.warn(
            "set_min_raise() is deprecated. Use _test_set_min_raise() for testing only. "
            "These methods should not be used in production/training code!",
            DeprecationWarning,
            stacklevel=2
        )
        self._test_set_min_raise(new_min_raise)
    
    def set_board(self, board_cards: List[int]) -> None:
        """DEPRECATED: Use _test_set_board() instead. This method will be removed."""
        import warnings
        warnings.warn(
            "set_board() is deprecated. Use _test_set_board() for testing only. "
            "These methods should not be used in production/training code!",
            DeprecationWarning,
            stacklevel=2
        )
        self._test_set_board(board_cards)
    
    def set_player_hole(self, player_idx: int, hole_cards: Optional[Tuple[int, int]]) -> None:
        """DEPRECATED: Use _test_set_player_hole() instead. This method will be removed."""
        import warnings
        warnings.warn(
            "set_player_hole() is deprecated. Use _test_set_player_hole() for testing only. "
            "These methods should not be used in production/training code!",
            DeprecationWarning,
            stacklevel=2
        )
        self._test_set_player_hole(player_idx, hole_cards)
    
    # Add abbreviated deprecated methods for the most commonly used ones
    def set_player_stack(self, player_idx: int, new_stack: int) -> None:
        """DEPRECATED: Use _test_set_player_stack() instead."""
        import warnings
        warnings.warn("set_player_stack() is deprecated. Use _test_set_player_stack() for testing only.", DeprecationWarning, stacklevel=2)
        self._test_set_player_stack(player_idx, new_stack)
    
    def set_card_scenario(self, board: List[int], hole_cards: List[Optional[Tuple[int, int]]], undealt: Optional[List[int]] = None) -> None:
        """DEPRECATED: Use _test_set_card_scenario() instead."""
        import warnings
        warnings.warn("set_card_scenario() is deprecated. Use _test_set_card_scenario() for testing only.", DeprecationWarning, stacklevel=2)
        self._test_set_card_scenario(board, hole_cards, undealt)
    
    def validate_and_fix_state(self, fix_issues: bool = False) -> List[str]:
        """DEPRECATED: Use _test_validate_and_fix_state() instead."""
        import warnings
        warnings.warn("validate_and_fix_state() is deprecated. Use _test_validate_and_fix_state() for testing only.", DeprecationWarning, stacklevel=2)
        return self._test_validate_and_fix_state(fix_issues)
