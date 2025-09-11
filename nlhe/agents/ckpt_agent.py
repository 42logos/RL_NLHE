from __future__ import annotations
from typing import Dict, Any
import numpy as np

from ..core.types import Action, ActionType, GameState
from .base import Agent, EngineLike

SENTINEL = -1

class CKPTAgent(Agent):
    """Agent that acts using a saved RLlib checkpoint."""
    def __init__(self, checkpoint_path: str, history_len: int = 64):
        from ray.rllib.algorithms.algorithm import Algorithm
        self.history_len = history_len
        # load algorithm from checkpoint
        self.algo = Algorithm.from_checkpoint(checkpoint_path)
        # put model into eval mode if possible
        try:
            policy = self.algo.get_policy()
            model = getattr(policy, "model", None)
            if model is not None:
                model.eval()
        except Exception:
            pass

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

    def _map_action(self, env: EngineLike, s: GameState, seat: int, a: Dict[str, Any]) -> Action:
        info = env.legal_actions(s)
        atype = int(a.get("atype", 1))
        r = float(a.get("r", 0.0))
        r = max(0.0, min(1.0, r))
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

    def act(self, env: EngineLike, s: GameState, seat: int) -> Action:
        obs = self._obs_from_state(s, seat)
        try:
            # Older stack: policy exposes compute_single_action
            policy = self.algo.get_policy()
            action, _ = policy.compute_single_action(obs, explore=False)
        except Exception:
            try:
                # New RLModule stack: run module.forward_inference manually
                module = self.algo.get_module()
                import torch
                def _flat(o):
                    parts = []
                    parts.append(np.asarray(o["board"], dtype=np.float32).reshape(-1))
                    bl = int(o["board_len"])
                    bl_onehot = np.zeros(6, dtype=np.float32); bl_onehot[bl] = 1.0
                    parts.append(bl_onehot)
                    for k in ["current_bet", "hero_bet", "hero_cont", "hero_stack", "pot"]:
                        parts.append(np.array([o[k]], dtype=np.float32))
                    parts.append(np.asarray(o["hero_hole"], dtype=np.float32).reshape(-1))
                    parts.append(np.asarray(o["history"], dtype=np.float32).reshape(-1))
                    return np.concatenate(parts, axis=0)
                obs_vec = _flat(obs)
                obs_t = torch.from_numpy(obs_vec[None, ...])
                out = module.forward_inference({"obs": obs_t})
                dist_cls = module.get_inference_action_dist_cls()
                logits = out["action_dist_inputs"]
                dist = dist_cls.from_logits(logits) if hasattr(dist_cls, "from_logits") else dist_cls(logits)
                act_t = dist.sample()
                if isinstance(act_t, dict):
                    action = {k: (v.item() if hasattr(v, "item") else v) for k, v in act_t.items()}
                else:
                    action = act_t.cpu().numpy()
            except Exception:
                action, _ = self.algo.compute_single_action(obs, explore=False)
        if isinstance(action, (np.ndarray, list)):
            a_dict = {"atype": int(action[0]), "r": float(action[1])}
        elif isinstance(action, dict):
            a_dict = {"atype": action.get("atype", 1), "r": action.get("r", 0.0)}
        else:
            raise RuntimeError("Unsupported action format from policy")
        return self._map_action(env, s, seat, a_dict)
