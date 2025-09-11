from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray import tune
import gymnasium as gym

from ..core.types import Action, ActionType, GameState
from .base import Agent, EngineLike
from ..train.flattener import NLHEFlattener
from ..envs.param_env import NLHEParamEnv  # your gym.Env
from ..train.flattener import FlattenObsWrapper  # the wrapper version for training

SENTINEL = -1


def _ensure_env_registered(env_id: str = "nlhe_flat") -> None:
    """
    Make sure Gym/RLlib knows how to construct `env_id` before Algorithm.from_checkpoint.
    We register a creator that wraps NLHEParamEnv with FlattenObsWrapper.
    """
    # If already in Gym registry, do nothing.
    try:
        # Gymnasium >=0.29
        if env_id in gym.envs.registry:
            return
    except Exception:
        pass
    # If RLlib/Tune already knows it, also fine—but registering twice is harmless.
    def make_env(env_config):
        # env_config is a dict-like EnvContext from RLlib
        hero_seat   = env_config.get("hero_seat", 0)
        bb          = env_config.get("bb", 2)
        sb          = env_config.get("sb", bb / 2)
        seed        = env_config.get("seed", 0)
        start_stack = env_config.get("start_stack", 100)
        history_len = env_config.get("history_len", 64)

        base = NLHEParamEnv(
            hero_seat=hero_seat,
            bb=bb,
            sb=sb,
            seed=seed,
            start_stack=start_stack,
            history_len=history_len,
        )
        return FlattenObsWrapper(base, history_len=history_len)

    tune.register_env(env_id, lambda cfg: make_env(cfg))


class CKPTAgent(Agent):
    """
    Agent that loads an RLlib checkpoint and acts via the RLModule (new API).
    - Auto-registers the training env ID (so from_checkpoint() can rebuild).
    - Reuses the SAME observation layout via NLHEFlattener.
    - Supports recurrent and feed-forward policies; keeps per-seat RNN state.
    - Deterministic by default.
    """

    def __init__(
        self,
        checkpoint_path: str,
        history_len: int = 64,
        deterministic: bool = True,
        env_id: str = "nlhe_flat",
    ):
        self.history_len = history_len
        self.deterministic = deterministic

        # 0) Ensure env used in training is registered BEFORE restoring.
        _ensure_env_registered(env_id)

        # 1) Restore Algorithm, then get the single-agent RLModule
        self.algo: Algorithm = Algorithm.from_checkpoint(checkpoint_path)
        self._module = self.algo.get_module() or self.algo.get_module("default_policy")
        if self._module is None:
            raise RuntimeError("Could not obtain RLModule from Algorithm (new RLlib API).")

        # 2) Device + eval mode
        try:
            self._device = next(self._module.parameters()).device
        except Exception:
            self._device = torch.device("cpu")
        try:
            self._module.eval()
        except Exception:
            pass

        # 3) Shared flattener (single source of truth)
        self.flattener = NLHEFlattener(history_len=history_len)

        # 4) Per-seat recurrent state cache
        self._rnn_state: Dict[int, List[torch.Tensor]] = {}

    # -------- Observation construction (env-native dict) --------
    def _obs_from_state(self, s: GameState, seat: int) -> Dict[str, Any]:
        p = s.players[seat]
        hole = p.hole if p.hole is not None else (SENTINEL, SENTINEL)
        board = s.board + [SENTINEL] * (5 - len(s.board))

        hist = s.actions_log[-self.history_len:]
        pad = self.history_len - len(hist)
        if pad > 0:
            hist = [(SENTINEL, SENTINEL, SENTINEL, SENTINEL)] * pad + hist

        return {
            "pot": np.int32(s.pot),
            "current_bet": np.int32(s.current_bet),
            "board": np.asarray(board, dtype=np.int32),
            "board_len": int(len(s.board)),
            "hero_stack": np.int32(p.stack),
            "hero_bet": np.int32(p.bet),
            "hero_cont": np.int32(p.cont),
            "hero_hole": np.asarray(hole, dtype=np.int32),
            "history": np.asarray(hist, dtype=np.int32),
        }

    # -------- Action mapping (legal-aware) --------
    def _map_action(self, env: EngineLike, s: GameState, seat: int, a: Dict[str, Any]) -> Action:
        info = env.legal_actions(s)
        atype = int(a.get("atype", 1))
        r = float(a.get("r", 0.0))
        r = max(0.0, min(1.0, r))  # clamp

        acts = getattr(info, "actions", [])
        owe = env.owed(s, seat)

        def has(kind: ActionType) -> bool:
            return any(x.kind == kind for x in acts)

        if atype == 0 and has(ActionType.FOLD) and owe > 0:
            return Action(ActionType.FOLD)
        if atype == 1 and has(ActionType.CHECK) and owe == 0:
            return Action(ActionType.CHECK)
        if atype == 2 and has(ActionType.CALL) and owe > 0:
            return Action(ActionType.CALL)
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

        # Fallbacks
        if owe == 0 and has(ActionType.CHECK):
            return Action(ActionType.CHECK)
        if owe > 0 and has(ActionType.CALL):
            return Action(ActionType.CALL)
        if owe > 0 and has(ActionType.FOLD):
            return Action(ActionType.FOLD)
        for x in acts:
            if x.kind == ActionType.RAISE_TO:
                return Action(ActionType.RAISE_TO, amount=getattr(info, "min_raise_to", s.current_bet + s.min_raise))
        return Action(ActionType.CHECK)

    # -------- Public API --------
    def reset_seat_state(self, seat: int) -> None:
        """Reset RNN state for a seat (e.g., on new episode)."""
        self._rnn_state.pop(seat, None)

    def act(self, env: EngineLike, s: GameState, seat: int) -> Action:
        # 1) Build raw observation and flatten to EXACT train/eval layout
        obs_raw = self._obs_from_state(s, seat)
        obs_flat = self.flattener.transform(obs_raw)  # np.float32 [F]
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=self._device).unsqueeze(0)  # [1, F]

        # 2) RNN state (+ SEQ_LENS)
        state_in: Optional[List[torch.Tensor]] = self._rnn_state.get(seat)
        if state_in is None:
            try:
                init = self._module.get_initial_state()
            except Exception:
                init = []
            state_in = []
            for x in init:
                t = torch.as_tensor(x, device=self._device)
                if t.ndim == 1:
                    t = t.unsqueeze(0)  # [1, H]
                state_in.append(t)

        batch = {
            Columns.OBS: obs_t,
            Columns.SEQ_LENS: torch.tensor([1], dtype=torch.int32, device=self._device),
        }
        if state_in:
            batch[Columns.STATE_IN] = state_in

        # 3) Inference via RLModule (new stack)
        with torch.no_grad():
            out = self._module.forward_inference(batch)

            if Columns.ACTIONS in out:
                act_t = out[Columns.ACTIONS]
            else:
                dist_cls = self._module.get_inference_action_dist_cls()
                dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
                if self.deterministic and hasattr(dist, "to_deterministic"):
                    dist = dist.to_deterministic()
                act_t = dist.sample()

            # Update per-seat RNN state
            self._rnn_state[seat] = out.get(Columns.STATE_OUT, state_in)

        # 4) Normalize to {"atype","r"} and map into engine action
        if isinstance(act_t, dict):
            atype_val = act_t.get("atype", 1)
            r_val = act_t.get("r", 0.0)
            atype = int(torch.as_tensor(atype_val).view(-1)[0].item())
            r = float(torch.as_tensor(r_val).view(-1)[0].item())
        else:
            arr = torch.as_tensor(act_t, device=self._device).view(-1)
            atype = int(arr[0].item())
            r = float(arr[1].item()) if arr.numel() > 1 else 0.0

        return self._map_action(env, s, seat, {"atype": atype, "r": r})
