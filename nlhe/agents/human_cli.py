from __future__ import annotations
from typing import List
from ..core.types import Action, ActionType, GameState
from .base import Agent, EngineLike

def _fmt_options(env: EngineLike, s: GameState, seat: int):
    info = env.legal_actions(s)
    acts: List[Action] = getattr(info, "actions", [])
    owe = env.owed(s, seat)
    options = []
    for a in acts:
        if a.kind == ActionType.FOLD: options.append("[F]old")
        elif a.kind == ActionType.CHECK: options.append("[K]heck")
        elif a.kind == ActionType.CALL: options.append(f"[C]all({owe})")
        elif a.kind == ActionType.RAISE_TO:
            min_to = getattr(info, "min_raise_to", s.current_bet)
            max_to = getattr(info, "max_raise_to", s.current_bet)
            has_rr = getattr(info, "has_raise_right", False)
            if has_rr: options.append(f"[R]aise to X (min {min_to}, max {max_to}) or 'allin'")
            else: options.append(f"All-in only (max {max_to}) — rights closed")
    return options, info

class HumanAgent(Agent):
    def act(self, env: EngineLike, s: GameState, seat: int) -> Action:
        assert s.next_to_act == seat
        options, info = _fmt_options(env, s, seat)
        print("Legal:", ", ".join(options) if options else "(none)")
        while True:
            raw = input("> ").strip().lower()
            acts: List[Action] = getattr(info, "actions", [])
            if raw in ("f","fold") and any(a.kind==ActionType.FOLD for a in acts):
                return Action(ActionType.FOLD)
            if raw in ("k","check") and any(a.kind==ActionType.CHECK for a in acts):
                return Action(ActionType.CHECK)
            if raw in ("c","call") and any(a.kind==ActionType.CALL for a in acts):
                return Action(ActionType.CALL)
            if any(a.kind==ActionType.RAISE_TO for a in acts):
                min_to = getattr(info, "min_raise_to", s.current_bet)
                max_to = getattr(info, "max_raise_to", s.current_bet)
                has_rr = getattr(info, "has_raise_right", False)
                if raw in ("ai","allin","all-in","all in"):
                    return Action(ActionType.RAISE_TO, amount=max_to)
                if raw.startswith("r ") or raw.startswith("raise ") or raw.startswith("bet "):
                    try:
                        amt = int(raw.split()[1])
                    except Exception:
                        print("Enter like: 'r 12' or 'raise 12'"); continue
                    if amt <= s.current_bet: print(f"Amount must exceed current bet ({s.current_bet})."); continue
                    if amt > max_to: print(f"Max is all-in {max_to}."); continue
                    if not has_rr and amt != max_to: print("Only all-in allowed (raise rights closed)."); continue
                    if has_rr and (amt < min_to) and (amt != max_to): print(f"Min raise-to is {min_to} (or all-in {max_to})."); continue
                    return Action(ActionType.RAISE_TO, amount=amt)
            print("Invalid input. Try: f/k/c, 'r X', or 'allin'.")
