# nlhe/obs/flattener.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

SENTINEL = -1

@dataclass
class NLHEFlattener:
    """Deterministically flatten NLHE obs dict -> np.float32 vector.
       Layout (size=274 when history_len=64):
         board[5] + onehot(board_len,6) + [current_bet, hero_bet, hero_cont, hero_stack, pot]
         + hero_hole[2] + history[history_len*4]
    """
    history_len: int = 64

    @property
    def size(self) -> int:
        return 5 + 6 + 5 + 2 + 4 * self.history_len

    def transform(self, obs: dict) -> np.ndarray:
        # --- board (pad to 5) ---
        board = np.asarray(obs.get("board", []), dtype=np.float32).reshape(-1)
        if board.size < 5:
            board = np.concatenate(
                [board, np.full(5 - board.size, float(SENTINEL), dtype=np.float32)]
            )
        elif board.size > 5:
            board = board[:5]

        # --- board_len onehot (6) ---
        bl = int(obs.get("board_len", 0))
        bl = max(0, min(5, bl))
        bl_onehot = np.zeros(6, dtype=np.float32)
        bl_onehot[bl] = 1.0

        # --- scalars ---
        scalars = []
        for k in ["current_bet", "hero_bet", "hero_cont", "hero_stack", "pot"]:
            v = np.array([obs.get(k, 0)], dtype=np.float32)
            scalars.append(v)

        # --- hero_hole (pad to 2) ---
        hole = np.asarray(obs.get("hero_hole", []), dtype=np.float32).reshape(-1)
        if hole.size < 2:
            hole = np.concatenate(
                [hole, np.full(2 - hole.size, float(SENTINEL), dtype=np.float32)]
            )
        elif hole.size > 2:
            hole = hole[:2]

        # --- history (history_len * 4) ---
        hist = np.asarray(obs.get("history", []), dtype=np.float32).reshape(-1)
        target = 4 * self.history_len
        if hist.size < target:
            pad = np.full(target - hist.size, float(SENTINEL), dtype=np.float32)
            hist = np.concatenate([pad, hist])
        elif hist.size > target:
            hist = hist[-target:]

        out = np.concatenate(
            [board, bl_onehot] + scalars + [hole, hist]
        ).astype(np.float32, copy=False)

        # Guardrail
        assert out.shape == (self.size,), f"flat obs size {out.shape} != {(self.size,)}"
        return out


class FlattenObsWrapper(gym.ObservationWrapper):
    """Wrap any NLHE env that returns your dict-observation and emit a flat Box."""
    def __init__(self, env: gym.Env, history_len: int = 64):
        super().__init__(env)
        self.f = NLHEFlattener(history_len=history_len)
        low = np.full((self.f.size,), -np.inf, dtype=np.float32)
        high = np.full((self.f.size,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return self.f.transform(obs)
