from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..core.rs_engine import NLHEngine
from ..core.types import Action, ActionType, GameState
from ..agents.tamed_random import TamedRandomAgent

class NLHEParamEnv:
    def __init__(self, hero_seat: int = 0, seed: int = 42,
                 sb: int = 1, bb: int = 2, start_stack: int = 100,
                 history_len: int = 64):
        self.rng = random.Random(seed)
        self.hero = hero_seat
        self.engine = NLHEngine(sb=sb, bb=bb, start_stack=start_stack, rng=self.rng)
        self.agents = [TamedRandomAgent(self.rng) for _ in range(self.engine.N)]
        self.agents[self.hero] = None  # type: ignore
        self._state: Optional[GameState] = None
        self.H = history_len
        self.observation_space = spaces.Dict({
            "pot": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "current_bet": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "board": spaces.Box(low=0, high=51, shape=(5,), dtype=np.int32),
            "board_len": spaces.Discrete(6),
            "hero_stack": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "hero_bet": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "hero_cont": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "hero_hole": spaces.Box(low=0, high=51, shape=(2,), dtype=np.int32),
            "history": spaces.Box(low=-1, high=10_000, shape=(self.H, 4), dtype=np.int32),
        })
        self.action_space = spaces.Dict({
            "atype": spaces.Discrete(4),
            "r": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def _obs(self) -> Dict[str, Any]:
        s = self._state; assert s is not None
        p = s.players[self.hero]
        hole = p.hole or (0, 1)
        board = s.board + [0]*(5-len(s.board))
        hist = s.actions_log[-self.H:]; pad = self.H - len(hist)
        if pad > 0: hist = [(-1, -1, -1, -1)]*pad + hist
        hist_arr = np.array(hist, dtype=np.int32)
        return {
            "pot": s.pot, "current_bet": s.current_bet, "board": np.array(board, dtype=np.int32),
            "board_len": len(s.board), "hero_stack": p.stack, "hero_bet": p.bet, "hero_cont": p.cont,
            "hero_hole": np.array(list(hole), dtype=np.int32), "history": hist_arr,
        }

    def _advance_others_until_hero_or_done(self) -> Tuple[bool, Optional[List[int]]]:
        s = self._state; assert s is not None
        done = False; rewards = None
        while not done and s.next_to_act is not None and s.next_to_act != self.hero:
            seat = s.next_to_act
            a = self.agents[seat].act(self.engine, s, seat)  # type: ignore
            s, done, rewards, _ = self.engine.step(s, a)
            self._state = s
        return done, rewards

    def _map_action(self, a: Dict[str, Any]) -> Action:
        s = self._state; assert s is not None
        info = self.engine.legal_actions(s)
        atype = int(a.get("atype", 1))
        rraw = a.get("r", 0.0)
        try:
            import numpy as _np
            r = float(_np.array(rraw).reshape(-1)[0])
        except Exception:
            r = float(rraw)
        r = min(max(r, 0.0), 1.0)

        acts = getattr(info, "actions", [])
        owe = self.engine.owed(s, self.hero)
        def has(k): return any(x.kind == k for x in acts)

        if atype == 0 and has(ActionType.FOLD) and owe > 0: return Action(ActionType.FOLD)
        if atype == 1 and has(ActionType.CHECK) and owe == 0: return Action(ActionType.CHECK)
        if atype == 2 and has(ActionType.CALL) and owe > 0: return Action(ActionType.CALL)
        if atype == 3 and any(x.kind == ActionType.RAISE_TO for x in acts):
            min_to = getattr(info, "min_raise_to", s.current_bet)
            max_to = getattr(info, "max_raise_to", s.current_bet)
            has_rr = getattr(info, "has_raise_right", False)
            if (not has_rr) or (max_to < min_to):
                return Action(ActionType.RAISE_TO, amount=max_to)
            target = int(round(min_to + r * (max_to - min_to)))
            target = max(min_to, min(target, max_to))
            if target <= s.current_bet: target = min_to
            return Action(ActionType.RAISE_TO, amount=target)

        if owe == 0 and has(ActionType.CHECK): return Action(ActionType.CHECK)
        if owe > 0 and has(ActionType.CALL): return Action(ActionType.CALL)
        if owe > 0 and has(ActionType.FOLD): return Action(ActionType.FOLD)
        for x in acts:
            if x.kind == ActionType.RAISE_TO:
                return Action(ActionType.RAISE_TO, amount=getattr(info, "min_raise_to", s.current_bet + s.min_raise))
        return Action(ActionType.CHECK)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None: self.rng.seed(seed)
        self._state = self.engine.reset_hand(button=0)
        done, rewards = self._advance_others_until_hero_or_done()
        obs = self._obs(); info = {}
        if done: info["terminal_rewards"] = rewards
        return obs, info

    def step(self, action: Dict[str, Any]):
        s = self._state; assert s is not None
        done, rewards = self._advance_others_until_hero_or_done()
        if done:
            obs = self._obs(); return obs, 0.0, True, False, {"terminal_rewards": rewards}
        assert s.next_to_act == self.hero
        a = self._map_action(action)
        s, done, rewards, _ = self.engine.step(s, a)
        self._state = s
        if not done: done, rewards = self._advance_others_until_hero_or_done()
        obs = self._obs()
        reward = float(rewards[self.hero]) if (done and rewards is not None) else 0.0
        return obs, reward, bool(done), False, {}
