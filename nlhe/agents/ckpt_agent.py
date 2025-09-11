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

    def _flatten_obs_like_connector(self, o: dict) -> np.ndarray:
        """Flatten observations to match FlattenObservations connector behavior."""
        parts = []
        parts.append(np.asarray(o["board"], dtype=np.float32).reshape(-1))          # 5
        # board_len as single value, not one-hot (the connector might not one-hot encode it)
        parts.append(np.array([o["board_len"]], dtype=np.float32))                  # 1
        for k in ["current_bet", "hero_bet", "hero_cont", "hero_stack", "pot"]:     # 5 × 1
            parts.append(np.array([o[k]], dtype=np.float32))
        parts.append(np.asarray(o["hero_hole"], dtype=np.float32).reshape(-1))      # 2
        parts.append(np.asarray(o["history"], dtype=np.float32).reshape(-1))        # 64*4 = 256
        return np.concatenate(parts, axis=0)                                        # shape = [D]

    def act(self, env: EngineLike, s: GameState, seat: int) -> Action:
        obs = self._obs_from_state(s, seat)
        
        # Use new RLModule API
        try:
            # Try new API stack first
            import torch
            module = self.algo.get_module()
            device = "cpu"  # Default to CPU
            if hasattr(module, "parameters"):
                try:
                    device = str(next(module.parameters()).device)  # type: ignore
                except StopIteration:
                    pass
            
            # Convert observation to tensor batch
            if isinstance(obs, dict):
                # Flatten the dictionary observation to match training connector behavior
                obs_flat = self._flatten_obs_like_connector(obs)
                obs_batch = torch.from_numpy(obs_flat).unsqueeze(0).to(device)  # Add batch dimension
            else:
                obs_batch = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0).to(device)
            
            # Forward inference
            with torch.no_grad():
                out = module.forward_inference({"obs": obs_batch})  # type: ignore
                
            # Extract action from output
            if "actions" in out:
                action = out["actions"][0]  # Remove batch dimension
            elif "action_dist_inputs" in out:
                # Sample from distribution
                dist_cls = module.get_inference_action_dist_cls()  # type: ignore
                logits = out["action_dist_inputs"]
                try:
                    if hasattr(dist_cls, "from_logits"):
                        dist = dist_cls.from_logits(logits)
                    else:
                        dist = dist_cls(logits)  # type: ignore
                    action = dist.sample()[0]  # Remove batch dimension
                except Exception:
                    # Fallback: just use the logits directly as action
                    action = logits[0]
            else:
                raise RuntimeError("No action or action_dist_inputs in module output")
                
            # Convert tensor to numpy if needed
            if hasattr(action, "cpu") and hasattr(action, "numpy"):
                action = action.cpu().numpy()  # type: ignore
                
        except (AttributeError, ImportError):
            # Fallback to old API
            try:
                policy = self.algo.get_policy()
                out = policy.compute_single_action(obs, explore=False)
            except AttributeError:
                out = self.algo.compute_single_action(obs, explore=False)
            action = out[0] if isinstance(out, (tuple, list)) else out
        
        # Process action format
        if isinstance(action, (np.ndarray, list)):
            a_dict = {"atype": int(action[0]), "r": float(action[1])}
        elif isinstance(action, dict):
            a_dict = {"atype": action.get("atype", 1), "r": action.get("r", 0.0)}
        else:
            raise RuntimeError("Unsupported action format from policy")
        return self._map_action(env, s, seat, a_dict)
