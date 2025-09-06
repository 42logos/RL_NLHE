from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..core.rs_engine import NLHEngine
from ..core.types import Action, ActionType, GameState
from ..agents.tamed_random import TamedRandomAgent

SENTINEL = -1  # 用 -1 作为未揭示/未持有的占位

class NLHEParamEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", None], "render_fps": 30}

    def __init__(self, 
             config: Optional[dict] = None,
             **kwargs):

        cfg = {}
        if config is not None:
            # EnvContext 基本上是 dict，直接用；保底转一下
            cfg.update(dict(config))
        if kwargs:
            cfg.update(kwargs)
            
        self.debug_contains_check = bool(cfg.get("debug_contains_check", False))
        self.render_mode = cfg.get("render_mode", None)

        hero_seat   = int(cfg.get("hero_seat", 0))
        seed        = cfg.get("seed", 42)
        sb          = int(cfg.get("sb", 1))
        bb          = int(cfg.get("bb", 2))
        start_stack = int(cfg.get("start_stack", 100))
        history_len = int(cfg.get("history_len", 64))

        # ---------- 原先的初始化逻辑 ----------
        self.rng = random.Random(seed if seed is not None else 42)
        self.hero = hero_seat
        self.engine = NLHEngine(sb=sb, bb=bb, start_stack=start_stack, rng=self.rng)
        self.agents = [TamedRandomAgent(self.rng) for _ in range(self.engine.N)]
        self.agents[self.hero] = None  # type: ignore
        self._state: Optional[GameState] = None
        self.H = history_len

        # 观测/动作空间 —— 保持你原来的定义（略）
        self.observation_space = spaces.Dict({
            "pot":          spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "current_bet":  spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "board":        spaces.Box(low=-1, high=51, shape=(5,), dtype=np.int32),
            "board_len":    spaces.Discrete(6),
            "hero_stack":   spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "hero_bet":     spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "hero_cont":    spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
            "hero_hole":    spaces.Box(low=-1, high=51, shape=(2,), dtype=np.int32),
            "history":      spaces.Box(low=-1, high=10_000, shape=(self.H, 4), dtype=np.int32),
        })
        self.action_space = spaces.Dict({
            "atype": spaces.Discrete(4),
            "r":     spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })

        # 可选：按钮位
        self._button = 0

    # ---------- 内部：构造观测（dataclass 保留在内部好用） ----------
    def _obs_struct(self) -> ParamEnv_Observation:
        s = self._state; assert s is not None
        p = s.players[self.hero]
        hole = p.hole if p.hole is not None else (SENTINEL, SENTINEL)
        board = s.board + [SENTINEL] * (5 - len(s.board))
        hist = s.actions_log[-self.H:]; pad = self.H - len(hist)
        if pad > 0:
            hist = [ (SENTINEL, SENTINEL, SENTINEL, SENTINEL) ] * pad + hist
        return ParamEnv_Observation(
            pot=int(s.pot),
            current_bet=int(s.current_bet),
            board=np.asarray(board, dtype=np.int32),
            board_len=len(s.board),
            hero_stack=int(p.stack),
            hero_bet=int(p.bet),
            hero_cont=int(p.cont),
            hero_hole=np.asarray(list(hole), dtype=np.int32),
            history=np.asarray(hist, dtype=np.int32),
        )

    def _obs(self) -> Dict[str, Any]:
        o = self._obs_struct()
        obs = {
            "pot":         np.int32(o.pot),
            "current_bet": np.int32(o.current_bet),
            "board":       np.asarray(o.board, dtype=np.int32),
            "board_len":   int(o.board_len),
            "hero_stack":  np.int32(o.hero_stack),
            "hero_bet":    np.int32(o.hero_bet),
            "hero_cont":   np.int32(o.hero_cont),
            "hero_hole":   np.asarray(o.hero_hole, dtype=np.int32),
            "history":     np.asarray(o.history, dtype=np.int32),
        }
        if self.debug_contains_check:
            assert self.observation_space.contains(obs), f"obs 不符合空间: { {k: (np.array(v).shape, np.array(v).dtype) for k,v in obs.items()} }"
        return obs

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
        r = float(a.get("r", 0.0))
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)

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
            if target <= s.current_bet:
                target = min_to
            return Action(ActionType.RAISE_TO, amount=target)

        if owe == 0 and has(ActionType.CHECK): return Action(ActionType.CHECK)
        if owe > 0 and has(ActionType.CALL):  return Action(ActionType.CALL)
        if owe > 0 and has(ActionType.FOLD):  return Action(ActionType.FOLD)
        for x in acts:
            if x.kind == ActionType.RAISE_TO:
                return Action(ActionType.RAISE_TO, amount=getattr(info, "min_raise_to", s.current_bet + s.min_raise))
        return Action(ActionType.CHECK)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 让 Gym 初始化 self.np_random（复现友好）
        super().reset(seed=seed)
        # 用 Gym 的 RNG 给 Python Random 播种
        seed_int = int(self.np_random.integers(0, 2**32 - 1))
        self.rng.seed(seed_int)

        # 轮转按钮（可选）
        self._button = (self._button) % self.engine.N

        self._state = self.engine.reset_hand(button=self._button)
        done, rewards = self._advance_others_until_hero_or_done()
        obs = self._obs()
        info: Dict[str, Any] = {}
        if done:
            info["terminal_rewards"] = rewards
        # 下一局按钮右移
        self._button = (self._button + 1) % self.engine.N
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        s = self._state; assert s is not None
        done, rewards = self._advance_others_until_hero_or_done()
        if done:
            obs = self._obs()
            reward = float(rewards[self.hero]) if rewards is not None else 0.0
            return obs, reward, True, False, {"terminal_rewards": rewards}

        assert s.next_to_act == self.hero
        a = self._map_action(action)
        s, done, rewards, _ = self.engine.step(s, a)
        self._state = s
        if not done:
            done, rewards = self._advance_others_until_hero_or_done()
        obs = self._obs()
        reward = float(rewards[self.hero]) if (done and rewards is not None) else 0.0
        info = {"terminal_rewards": rewards} if done else {}
        return obs, reward, bool(done), False, info

    def render(self):
        if self.render_mode == "human":
            print(self._obs_struct())
        elif self.render_mode == "ansi":
            return str(self._obs_struct())

    def close(self):
        pass


@dataclass
class ParamEnv_Observation:
    pot: int
    current_bet: int
    board: np.ndarray
    board_len: int
    hero_stack: int
    hero_bet: int
    hero_cont: int
    hero_hole: np.ndarray
    history: np.ndarray

    # 可选：保留“像字典一样”的访问，但不直接返回给 Gym
    def __getitem__(self, k: str) -> Any: return getattr(self, k)
    def __iter__(self) -> Iterator[str]:
        return iter(("pot","current_bet","board","board_len",
                     "hero_stack","hero_bet","hero_cont","hero_hole","history"))
    def __len__(self) -> int: return 9
