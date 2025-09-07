"""Utilities for canonicalizing engine states for cross-engine comparison.

This module provides helper functions to map :class:`GameState` objects into
simple data structures made of built-in Python types.  The representation is
stable and hashable which makes it convenient for equality checks between
states produced by different engine implementations (e.g. the pure Python
engine and the Rust backend).

The main entry point is :func:`canonical_state` which converts a ``GameState``
into a nested tuple.  Two states that are logically equivalent will map to the
same canonical value even if they originate from different engine
implementations.
"""

from __future__ import annotations

from typing import Any, Tuple

from .types import GameState, PlayerState


def _player_to_tuple(p: PlayerState, include_cards: bool) -> Tuple[Any, ...]:
    """Convert a :class:`PlayerState` into a tuple of its public attributes."""
    hole = p.hole if include_cards else None
    return (hole, p.stack, p.bet, p.cont, p.status, p.rho)


def canonical_state(s: GameState, *, include_cards: bool = True) -> Tuple[Any, ...]:
    """Return a hashable canonical representation of ``s``.

    Parameters
    ----------
    s:
        The state to convert.
    include_cards:
        If ``True`` (default) the representation includes hole cards and the
        remaining deck.  Set to ``False`` to ignore card identities when only
        the betting state matters.
    """

    players = tuple(_player_to_tuple(p, include_cards) for p in s.players)
    board = tuple(s.board) if include_cards else ()
    undealt = tuple(s.undealt) if include_cards else ()
    return (
        s.button,
        s.round_label,
        board,
        undealt,
        players,
        s.current_bet,
        s.min_raise,
        s.tau,
        s.next_to_act,
        s.step_idx,
        s.pot,
        s.sb,
        s.bb,
        tuple(tuple(log) for log in s.actions_log),
    )


def states_equal(a: GameState, b: GameState) -> bool:
    """Determine if two ``GameState`` objects are equivalent.

    Parameters
    ----------
    a, b:
        The states to compare.

    Returns
    -------
    bool
        ``True`` if the states map to the same canonical representation,
        ``False`` otherwise.
    """

    return canonical_state(a) == canonical_state(b)
