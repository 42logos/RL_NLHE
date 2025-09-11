import random
from itertools import combinations
from typing import List

from nlhe.core.engine import NLHEngine as _PyEngine
from nlhe.core.types import Action, ActionType, LegalActionInfo


def best5_rank_from_7_py(cards: List[int]):
    from nlhe.core.eval import hand_rank_5
    best = (-1, ())
    for c5 in combinations(cards, 5):
        r = hand_rank_5(tuple(c5))
        if r > best:
            best = r
    return best

_ACTION_ID = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.RAISE_TO: 3,
}


def legal_actions_bits_now() -> tuple[int, int | None, int | None, bool | None]:
    """Module level placeholder to satisfy rs_engine import checks."""
    raise RuntimeError("Use NLHEngine.legal_actions_bits_now on an instance")


class NLHEngine(_PyEngine):
    def __init__(
        self,
        sb: int = 1,
        bb: int = 2,
        start_stack: int = 100,
        num_players: int = 6,
        seed: int | None = None,
    ):
        rng = random.Random(seed) if seed is not None else None
        super().__init__(
            sb=sb, bb=bb, start_stack=start_stack, num_players=num_players, rng=rng
        )
        self._state = None

    def reset_hand(self, button: int = 0):
        self._state = super().reset_hand(button)
        return self._state

    def reset_hand_apply_py(self, s, button: int = 0):
        new_state = super().reset_hand(button)
        s.__dict__.update(new_state.__dict__)
        self._state = s
        return s

    def step_apply_py_raw(self, s, action_id: int, amount):
        owe = self.owed(s, s.next_to_act)
        if action_id == 0:
            a = Action(ActionType.FOLD)
        elif action_id == 1:
            a = Action(ActionType.CHECK)
        elif action_id == 2:
            a = Action(ActionType.CALL if owe > 0 else ActionType.CHECK)
        elif action_id == 3:
            a = Action(ActionType.RAISE_TO, amount=amount)
        else:
            raise ValueError("invalid action id")
        new_state, done, rewards, _ = self.step(s, a)
        s.__dict__.update(new_state.__dict__)
        self._state = s
        return done, rewards

    def advance_round_if_needed_apply_py(self, s):
        done, rewards = self.advance_round_if_needed(s)
        self._state = s
        return done, rewards

    def legal_actions_bits_now(self) -> tuple[int, int | None, int | None, bool | None]:
        assert self._state is not None, "state not initialized"
        info: LegalActionInfo = self.legal_actions(self._state)
        mask = 0
        for act in info.actions:
            mask |= 1 << _ACTION_ID[act.kind]
        return (
            mask,
            info.min_raise_to,
            info.max_raise_to,
            info.has_raise_right,
        )
