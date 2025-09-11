from __future__ import annotations
from typing import Protocol
from ..core.types import Action, LegalActionInfo
from ..core.types import GameState

class Agent(Protocol):
    def act(self, env: "EngineLike", s: GameState, seat: int) -> Action: ...

class EngineLike(Protocol):
    def legal_actions(self, s: GameState) -> LegalActionInfo: ...
    def owed(self, s: GameState, i: int) -> int: ...
