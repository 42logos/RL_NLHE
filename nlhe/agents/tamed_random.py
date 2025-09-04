from __future__ import annotations
import random
from typing import Optional, List
from ..core.types import Action, ActionType, GameState
from .base import Agent, EngineLike

class TamedRandomAgent(Agent):
    def __init__(self, rng: Optional[random.Random] = None,
                 p_allin: float = 0.01, p_raise: float = 0.15,
                 p_raise_closed: float = 0.02, cap_raises_bb: int = 4):
        self.rng = rng or random.Random()
        self.p_allin = p_allin
        self.p_raise = p_raise
        self.p_raise_closed = p_raise_closed
        self.cap_raises_bb = cap_raises_bb

    def act(self, env: EngineLike, s: GameState, seat: int) -> Action:
        info = env.legal_actions(s)
        acts: List[Action] = getattr(info, "actions", []) if hasattr(info, "actions") else info.get("actions", [])
        if not acts: return Action(ActionType.CHECK)
        owe = env.owed(s, seat)

        def has(kind: ActionType) -> bool:
            return any(a.kind == kind for a in acts)

        def short_only_or_closed(min_to: int, max_to: int, has_rr: bool) -> bool:
            return (not has_rr) or (max_to < min_to)

        if owe == 0:
            if has(ActionType.CHECK) and self.rng.random() > self.p_raise:
                return Action(ActionType.CHECK)
            for a in acts:
                if a.kind == ActionType.RAISE_TO:
                    min_to = getattr(info, "min_raise_to", s.current_bet)
                    max_to = getattr(info, "max_raise_to", s.current_bet)
                    has_rr = getattr(info, "has_raise_right", False)
                    if short_only_or_closed(min_to, max_to, has_rr):
                        if self.rng.random() < self.p_raise_closed:
                            return Action(ActionType.RAISE_TO, amount=max_to)
                        return Action(ActionType.CHECK) if has(ActionType.CHECK) else (Action(ActionType.CALL) if has(ActionType.CALL) else acts[0])
                    cap = min(max_to, max(min_to, s.current_bet) + self.cap_raises_bb * s.bb)
                    cap = max(cap, min_to)
                    target = min_to if (self.rng.random() < 0.8 or cap == min_to) else self.rng.randint(min_to, cap)
                    return Action(ActionType.RAISE_TO, amount=target)
            return Action(ActionType.CHECK) if has(ActionType.CHECK) else acts[0]
        else:
            if has(ActionType.CALL) and self.rng.random() > self.p_raise:
                return Action(ActionType.CALL)
            for a in acts:
                if a.kind == ActionType.RAISE_TO:
                    min_to = getattr(info, "min_raise_to", s.current_bet)
                    max_to = getattr(info, "max_raise_to", s.current_bet)
                    has_rr = getattr(info, "has_raise_right", False)
                    if short_only_or_closed(min_to, max_to, has_rr):
                        if self.rng.random() < self.p_raise_closed:
                            return Action(ActionType.RAISE_TO, amount=max_to)
                        if has(ActionType.CALL): return Action(ActionType.CALL)
                        if has(ActionType.FOLD): return Action(ActionType.FOLD)
                        break
                    cap = min(max_to, s.current_bet + self.cap_raises_bb * s.bb)
                    if cap <= s.current_bet:
                        return Action(ActionType.CALL) if has(ActionType.CALL) else (Action(ActionType.FOLD) if has(ActionType.FOLD) else acts[0])
                    if self.rng.random() < self.p_allin:
                        return Action(ActionType.RAISE_TO, amount=max_to)
                    target = min_to if self.rng.random() < 0.8 else max(min_to, min(cap, max_to))
                    return Action(ActionType.RAISE_TO, amount=target)
            if has(ActionType.CALL): return Action(ActionType.CALL)
            if has(ActionType.FOLD): return Action(ActionType.FOLD)
            return acts[0]
