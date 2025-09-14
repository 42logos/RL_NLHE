from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..core.types import Action, GameState, LegalActionInfo


class Agent(ABC):
    """Abstract base class for all poker agents."""

    @abstractmethod
    def act(self, engine: "EngineLike", state: GameState, seat: int) -> Action:
        """Select an action for ``seat`` given the current ``state``."""
        raise NotImplementedError


class EngineLike(Protocol):
    """Subset of :class:`NLHEngine` used by agents for type checking."""

    def legal_actions(self, state: GameState) -> LegalActionInfo:
        ...

    def owed(self, state: GameState, seat: int) -> int:
        ...
